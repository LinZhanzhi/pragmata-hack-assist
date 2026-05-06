"""Real-time hack-panel assist: select a game window, hot-key triggers a
plan-and-walk along the optimal path.

Trigger combo (while the chosen game window is foreground):
    RIGHT MOUSE held + MOUSE BUTTON 4 (XButton1) held + tap T

On trigger, the app captures the game window, finds the panel via YOLO,
warps it, classifies cells, runs the reward-aware best-first path, and
moves the OS cursor through each cell-center on screen so the in-game
range dot walks the path.

Tested on Windows 10/11. Requires the project's existing weights:
    panel_detector/runs/panel_detector/weights/best.pt   (YOLO)
    panel_detector/runs/cell_classifier/v1_best.pt       (cell classifier)
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import threading
import time
import traceback
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import win32con
import win32gui
import win32ui
from PIL import Image
from pynput import keyboard, mouse
from torchvision import transforms

# --- project imports ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent
PD = ROOT / "panel_detector"
sys.path.insert(0, str(PD / "scripts"))
from current_detector import (  # noqa: E402
    detect_orange_current,
    find_current_cell,
    find_orange_centroid_in_cell,
)
from dest_detector import infer_grid_from_dest  # noqa: E402
from path_demo import (  # noqa: E402
    DEST_CLASSES, FORBIDDEN_CLASSES, NORMAL_CLASSES, draw_overlay, solve,
)
from predict import (  # noqa: E402
    DEFAULT_WEIGHTS as YOLO_WEIGHTS,
    corners_from_bbox, warp_size_from_bbox,
)
from train_cell_classifier import IMG_SIZE, SmallCNN  # noqa: E402

CLS_WEIGHTS = PD / "runs" / "cell_classifier" / "v1_best.pt"
LIVE_DIR = PD / "runs" / "live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR = PD / "runs" / "debug"     # one subfolder per trigger when --debug

ALPHA, BETA = 5.0, 1.0
# Closed-loop walker tunables. We track the orange dot in panel space and
# nudge the mouse one screen-axis at a time. Step is fixed for now; the
# adaptive variant is preserved below in comments for later experiments.
WALK_TICK_S = 0.033          # post-move settle (~2 frames @ 60 FPS) so the
                              # captured frame reflects the input we just sent
WALK_STEP_PX = 120           # default OS-pixel UPPER BOUND per nudge (overridable via --step)
WALK_DAMPING = 0.85          # scale J_inv@err by this so calibration error
                              # doesn't translate into overshoot ping-pong
# WALK_STEP_INIT_PX = 25     # initial OS-pixel magnitude per nudge
# WALK_STEP_MIN_PX = 4
# WALK_STEP_MAX_PX = 200
# WALK_NO_PROGRESS_GROW = 6  # iterations w/ no cell change before growing step
WALK_MAX_TICKS_PER_HOP = 400 # ~16s per cell at 40ms tick


# --- Win32 helpers -----------------------------------------------------------
user32 = ctypes.WinDLL("user32", use_last_error=True)


def list_visible_windows():
    items = []

    def _enum(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if not title:
            return
        rect = win32gui.GetWindowRect(hwnd)
        if (rect[2] - rect[0]) < 100 or (rect[3] - rect[1]) < 100:
            return
        items.append((hwnd, title))

    win32gui.EnumWindows(_enum, None)
    return items


def capture_window_client(hwnd) -> np.ndarray | None:
    """Capture the client area of `hwnd` as BGR numpy array."""
    try:
        l, t, r, b = win32gui.GetClientRect(hwnd)
    except Exception:
        return None
    w, h = r - l, b - t
    if w <= 0 or h <= 0:
        return None

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(mfcDC, w, h)
    saveDC.SelectObject(bmp)
    # PW_RENDERFULLCONTENT (0x02) helps with DWM-composited / GPU windows.
    result = ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0x02)
    if not result:
        ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0x00)

    bmpinfo = bmp.GetInfo()
    bmpstr = bmp.GetBitmapBits(True)
    img = np.frombuffer(bmpstr, dtype=np.uint8).reshape(
        (bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4))[:, :, :3].copy()  # BGR

    win32gui.DeleteObject(bmp.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)
    return img


def client_to_screen(hwnd, x, y):
    pt = wintypes.POINT(int(x), int(y))
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y


# --- SendInput mouse move ----------------------------------------------------
# Use SendInput with absolute screen coords (normalized to 0..65535 over the
# virtual desktop) so games that ignore SetCursorPos still see the move.
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
INPUT_MOUSE = 0


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUT_UNION)]


def move_cursor_abs(screen_x, screen_y):
    vx = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    vy = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    vw = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    vh = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    nx = int(round((screen_x - vx) * 65535 / max(vw - 1, 1)))
    ny = int(round((screen_y - vy) * 65535 / max(vh - 1, 1)))
    mi = MOUSEINPUT(nx, ny, 0,
                    MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
                    0, None)
    inp = INPUT(INPUT_MOUSE, _INPUT_UNION(mi=mi))
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def move_cursor_rel(dx, dy):
    """Relative mouse delta (no ABSOLUTE flag). The game's own sensitivity
    determines how far the in-game aim moves per pixel of dx/dy; we only
    decide direction + magnitude in OS units."""
    mi = MOUSEINPUT(int(round(dx)), int(round(dy)), 0,
                    MOUSEEVENTF_MOVE, 0, None)
    inp = INPUT(INPUT_MOUSE, _INPUT_UNION(mi=mi))
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


# --- vision pipeline ---------------------------------------------------------
@dataclass
class PlanResult:
    panel_bgr: np.ndarray
    overlay_bgr: np.ndarray
    M_inv: np.ndarray            # 3x3 panel->client homography
    n_rows: int
    n_cols: int
    start: tuple
    goal: tuple
    path: list | None
    rewards: int
    expansions: int
    cell_w: float
    cell_h: float


@dataclass
class PanelView:
    panel_bgr: np.ndarray
    M_inv: np.ndarray            # 3x3 panel->client homography for this frame
    xyxy: np.ndarray
    kpts: np.ndarray
    conf: float


class Engine:
    def __init__(self):
        from ultralytics import YOLO
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[engine] device={self.device}")
        print(f"[engine] loading YOLO {YOLO_WEIGHTS}")
        self.yolo = YOLO(str(YOLO_WEIGHTS))
        print(f"[engine] loading classifier {CLS_WEIGHTS}")
        ckpt = torch.load(CLS_WEIGHTS, map_location=self.device,
                          weights_only=False)
        self.classes: list[str] = ckpt["classes"]
        self.cls_model = SmallCNN(len(self.classes)).to(self.device)
        self.cls_model.load_state_dict(ckpt["state_dict"])
        self.cls_model.eval()
        self.tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def detect_and_warp_panel(
        self,
        client_bgr: np.ndarray,
        out_size: tuple[int, int] | None = None,
    ) -> PanelView | None:
        """Detect the panel in ``client_bgr`` and warp it.

        If ``out_size`` is provided, the panel is warped to that fixed
        canonical size (width, height). This lets the live walker re-detect
        the panel on every frame while keeping a stable panel coordinate
        system for the path grid and Jacobian calibration.
        """
        results = self.yolo.predict(source=client_bgr, conf=0.25, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes
        best = int(boxes.conf.argmax().item())
        xyxy = boxes.xyxy[best].cpu().numpy()
        kpts = corners_from_bbox(xyxy)  # 4x2 in client coords (TL,TR,BR,BL)
        conf = float(boxes.conf[best])

        if out_size is None:
            out_w, out_h = warp_size_from_bbox(xyxy)
        else:
            out_w, out_h = (int(out_size[0]), int(out_size[1]))

        dst = np.array([[0, 0], [out_w - 1, 0],
                        [out_w - 1, out_h - 1], [0, out_h - 1]],
                       dtype=np.float32)
        M = cv2.getPerspectiveTransform(kpts, dst)
        M_inv = np.linalg.inv(M)
        panel = cv2.warpPerspective(client_bgr, M, (out_w, out_h))
        return PanelView(
            panel_bgr=panel,
            M_inv=M_inv,
            xyxy=xyxy,
            kpts=kpts,
            conf=conf,
        )

    def plan(self, client_bgr: np.ndarray,
             debug_dir: "Path | None" = None) -> "PlanResult | None":
        # 1. detect panel
        panel_view = self.detect_and_warp_panel(client_bgr)
        if panel_view is None:
            print("[plan] no panel detected")
            if debug_dir is not None:
                cv2.imwrite(str(debug_dir / "01_capture.png"), client_bgr)
            return None
        xyxy = panel_view.xyxy
        kpts = panel_view.kpts

        if debug_dir is not None:
            cv2.imwrite(str(debug_dir / "01_capture.png"), client_bgr)
            vis = client_bgr.copy()
            x0, y0, x1, y1 = (int(round(v)) for v in xyxy[:4])
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            corner_colors = [(0, 0, 255), (0, 255, 255),
                             (255, 0, 255), (255, 255, 0)]  # TL TR BR BL
            for (cx, cy), col in zip(kpts, corner_colors):
                cv2.circle(vis, (int(round(cx)), int(round(cy))), 6, col, -1)
            cv2.putText(vis, f"conf={panel_view.conf:.2f}",
                        (x0, max(0, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
            cv2.imwrite(str(debug_dir / "02_capture_with_bbox.png"), vis)

        panel = panel_view.panel_bgr
        M_inv = panel_view.M_inv

        if debug_dir is not None:
            cv2.imwrite(str(debug_dir / "03_panel_warped.png"), panel)

        # 2. infer grid + dest
        grid = infer_grid_from_dest(panel)
        if grid is None:
            print("[plan] no destination icon found in panel")
            return None
        n_rows, n_cols = grid["n_rows"], grid["n_cols"]
        goal = (grid["dest_row"], grid["dest_col"])

        # 3. find current cell (cyan ring)
        start = find_current_cell(panel, n_rows, n_cols, exclude=goal)
        if start is None:
            print("[plan] no current-location ring found")
            return None

        # 4. classify cells
        H, W = panel.shape[:2]
        xs = [round(c * W / n_cols) for c in range(n_cols + 1)]
        ys = [round(r * H / n_rows) for r in range(n_rows + 1)]
        crops = []
        cell_imgs = []
        for r in range(n_rows):
            for c in range(n_cols):
                cell = panel[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
                cell_imgs.append(cell)
                rgb = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
                crops.append(self.tf(Image.fromarray(rgb)))
        batch = torch.stack(crops).to(self.device)
        with torch.no_grad():
            probs = F.softmax(self.cls_model(batch), dim=1).cpu().numpy()
        labels = np.empty((n_rows, n_cols), dtype=object)
        for i in range(n_rows * n_cols):
            r, c = divmod(i, n_cols)
            labels[r, c] = self.classes[int(probs[i].argmax())]

        if debug_dir is not None:
            cells_dir = debug_dir / "04_cells"
            cells_dir.mkdir(exist_ok=True)
            for i, cell in enumerate(cell_imgs):
                r, c = divmod(i, n_cols)
                conf = float(probs[i].max())
                lbl = labels[r, c]
                fname = f"r{r:02d}_c{c:02d}_{lbl}_{conf:.2f}.png"
                cv2.imwrite(str(cells_dir / fname), cell)

        # 5. build kind grid
        kind = [["normal"] * n_cols for _ in range(n_rows)]
        for r in range(n_rows):
            for c in range(n_cols):
                lbl = labels[r, c]
                if (r, c) == goal or (r, c) == start:
                    kind[r][c] = "normal"
                elif lbl in FORBIDDEN_CLASSES:
                    kind[r][c] = "forbidden"
                elif lbl in NORMAL_CLASSES or lbl in DEST_CLASSES:
                    kind[r][c] = "normal"
                else:
                    kind[r][c] = "reward"

        path, rewards, expansions = solve(kind, start, goal,
                                          alpha=ALPHA, beta=BETA,
                                          max_expansions=20000)
        overlay = draw_overlay(panel, n_rows, n_cols, labels, path, start, goal)
        return PlanResult(
            panel_bgr=panel, overlay_bgr=overlay, M_inv=M_inv,
            n_rows=n_rows, n_cols=n_cols, start=start, goal=goal,
            path=path, rewards=rewards, expansions=expansions,
            cell_w=W / n_cols, cell_h=H / n_rows,
        )


# --- runtime app -------------------------------------------------------------
class AssistApp:
    def __init__(self, step_px: float = WALK_STEP_PX, tick_s: float = WALK_TICK_S,
                 debug: bool = False):
        self.engine: Engine | None = None
        self.target_hwnd: int | None = None
        self.target_title: str = ""
        self.rmb_down = False
        self.x1_down = False
        self.armed = True
        self.busy = False
        self.abort_walk = False
        self.event_q: Queue = Queue()
        self.step_px = float(step_px)
        self.tick_s = float(tick_s)
        self.debug = bool(debug)

    # ---- input listeners ----
    def _on_click(self, x, y, button, pressed):
        if button == mouse.Button.right:
            self.rmb_down = pressed
        elif button == mouse.Button.x1:
            self.x1_down = pressed

    def _on_key(self, key):
        # ESC always aborts an in-progress walk.
        if key == keyboard.Key.esc:
            if self.busy:
                self.abort_walk = True
            return
        if not self.armed or self.busy:
            return
        try:
            ch = key.char
        except AttributeError:
            return
        if ch and ch.lower() == "t" and self.rmb_down and self.x1_down:
            self.event_q.put("TRIGGER")

    def start_listeners(self):
        self._mouse_l = mouse.Listener(on_click=self._on_click)
        self._kbd_l = keyboard.Listener(on_press=self._on_key)
        self._mouse_l.start()
        self._kbd_l.start()

    def _write_walk_trace_debug(
        self,
        debug_dir: Path,
        plan: PlanResult,
        trace_points: list[tuple[float, float, bool]],
        end_reason: str,
    ) -> None:
        """Write one debug image showing planned path vs observed walk trace.

        The base image is the planned overlay (already in the canonical
        warped-panel coordinate system). On top of it we draw the observed
        orange-dot positions from each captured frame during the walk.

        Direct orange detections are drawn as orange circles; fallback
        positions (cell centers used when the orange dot was not detected)
        are drawn as yellow X markers so detection dropouts are visible.
        """
        canvas = plan.overlay_bgr.copy()

        if trace_points:
            pts = np.array([(x, y) for x, y, _direct in trace_points],
                           dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts], False, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.polylines(canvas, [pts], False, (0, 165, 255), 2, cv2.LINE_AA)

            for i, (x, y, direct) in enumerate(trace_points):
                px = int(round(x))
                py = int(round(y))
                if direct:
                    cv2.circle(canvas, (px, py), 4, (0, 165, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 6, (0, 0, 0), 1, cv2.LINE_AA)
                else:
                    cv2.line(canvas, (px - 4, py - 4), (px + 4, py + 4),
                             (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.line(canvas, (px - 4, py + 4), (px + 4, py - 4),
                             (0, 255, 255), 2, cv2.LINE_AA)
                if i % 5 == 0:
                    cv2.putText(canvas, str(i), (px + 6, py - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (255, 255, 255), 1, cv2.LINE_AA)

            fx = int(round(trace_points[-1][0]))
            fy = int(round(trace_points[-1][1]))
            cv2.circle(canvas, (fx, fy), 8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.rectangle(canvas, (8, 8), (520, 58), (0, 0, 0), -1)
        cv2.putText(canvas, "planned path + observed orange-dot trace",
                    (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"samples={len(trace_points)}  end={end_reason}",
                    (16, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imwrite(str(debug_dir / "06_walk_trace.png"), canvas)

    # ---- core: plan and walk ----
    def trigger(self):
        if self.busy or self.engine is None or self.target_hwnd is None:
            return
        self.busy = True
        try:
            t0 = time.time()
            self.abort_walk = False
            client = capture_window_client(self.target_hwnd)
            if client is None:
                print("[trigger] capture failed")
                return
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(LIVE_DIR / f"{ts}_capture.png"), client)
            debug_dir = None
            if self.debug:
                debug_dir = DEBUG_DIR / ts
                debug_dir.mkdir(parents=True, exist_ok=True)
                print(f"[debug] writing artifacts to {debug_dir}")
            plan = self.engine.plan(client, debug_dir=debug_dir)
            if plan is None:
                return
            cv2.imwrite(str(LIVE_DIR / f"{ts}_overlay.png"), plan.overlay_bgr)
            if debug_dir is not None:
                cv2.imwrite(str(debug_dir / "05_overlay.png"), plan.overlay_bgr)
            print(f"[trigger] plan: {plan.n_rows}x{plan.n_cols} "
                  f"start={plan.start} goal={plan.goal} "
                  f"path_len={(len(plan.path)-1) if plan.path else -1} "
                  f"rewards={plan.rewards} ({(time.time()-t0)*1000:.0f} ms)")
            if not plan.path:
                return
            self.walk_path(plan, debug_dir=debug_dir)
        except Exception:
            traceback.print_exc()
        finally:
            self.busy = False

    def _capture_panel_view(self, W, H) -> PanelView | None:
        """Capture a fresh game frame, re-detect the panel in that frame,
        and warp it into the fixed canonical panel size ``(W, H)``.

        This deliberately avoids using a saved transform across frames. If
        the panel slides or scales while the walk is running, the next step
        still reasons about a freshly re-locked panel view.
        """
        client = capture_window_client(self.target_hwnd)
        if client is None:
            return None
        if self.engine is None:
            return None
        return self.engine.detect_and_warp_panel(client, out_size=(W, H))

    def _measure_orange_panel(self, W, H):
        """Capture a fresh frame, re-detect + re-warp the panel, then
        return the orange-dot centroid in canonical panel pixels.

        Searches the WHOLE panel (not gated to a cell) so it works during
        calibration when the cyan ring may lag.
        """
        panel_view = self._capture_panel_view(W, H)
        if panel_view is None:
            return None
        info = detect_orange_current(panel_view.panel_bgr)
        if info is None:
            return None
        return float(info["center"][0]), float(info["center"][1])

    def _calibrate_jacobian(self, W, H, plan):
        """Probe the mouse on two screen directions, measure how the
        orange dot moves in panel pixels, return the 2x2 Jacobian
        ``J`` such that  panel_delta = J @ os_delta (panel_px / OS_px).

        The two probe directions are chosen from the planned path so the
        first probe nudges the cursor in the SAFE direction the player
        already wants to go (= toward path[1]), not blindly +X. This
        avoids the case where the panel forbids stepping right from
        the start cell. The second probe is perpendicular, with sign
        also taken from the path's first perpendicular hop when present
        (else +1).

        Returns None if the probes can't be observed.
        """
        PROBE_OS = 60.0
        # Settle longer than steady-state ticks: the very first probe
        # also has to wait for the capture pipeline to reflect the move.
        # Pragmata at 60 FPS needs at least ~33 ms; we double that.
        SETTLE = 0.07

        def measure(retries=3):
            for _ in range(retries):
                p = self._measure_orange_panel(W, H)
                if p is not None:
                    return p
                time.sleep(0.02)
            return None

        # ---- decide probe directions from the path geometry ----
        def panel_dir_to_screen(dr: float, dc: float) -> tuple[str, float]:
            """Geometric guess (via M_inv) at which screen axis & sign
            most closely produces the requested (dr, dc) panel motion.
            The game may invert this sign in practice; we only use it
            as a hint and trust the measured Jacobian afterward."""
            dx_p = dc * plan.cell_w
            dy_p = dr * plan.cell_h
            o = plan.M_inv @ np.array([W * 0.5, H * 0.5, 1.0])
            t = plan.M_inv @ np.array([W * 0.5 + dx_p,
                                       H * 0.5 + dy_p, 1.0])
            o /= o[2]; t /= t[2]
            sx = float(t[0] - o[0])
            sy = float(t[1] - o[1])
            if abs(sx) >= abs(sy):
                return ("h", 1.0 if sx > 0 else -1.0)
            return ("v", 1.0 if sy > 0 else -1.0)

        path = plan.path or []
        if len(path) >= 2:
            dr1 = path[1][0] - path[0][0]
            dc1 = path[1][1] - path[0][1]
            axis1, sign1 = panel_dir_to_screen(dr1, dc1)
        else:
            axis1, sign1 = "h", 1.0

        # Perpendicular axis; sign from first path hop that uses the
        # perpendicular panel axis. Default +1 when path is straight.
        axis2 = "v" if axis1 == "h" else "h"
        sign2 = 1.0
        perp_panel_is_y = (axis1 == "h")  # if first probe was screen-X
        # (which mostly maps to panel-X) then the perpendicular is panel-Y.
        for i in range(1, len(path) - 1):
            ddr = path[i + 1][0] - path[i][0]
            ddc = path[i + 1][1] - path[i][1]
            if perp_panel_is_y and ddr != 0:
                a, s = panel_dir_to_screen(ddr, 0)
                if a == axis2:
                    sign2 = s
                    break
            elif (not perp_panel_is_y) and ddc != 0:
                a, s = panel_dir_to_screen(0, ddc)
                if a == axis2:
                    sign2 = s
                    break

        print(f"[cal] probe1 {axis1}{'+' if sign1 > 0 else '-'} "
              f"probe2 {axis2}{'+' if sign2 > 0 else '-'} "
              f"(from path-first direction)")

        def probe_vec(axis: str, sign: float) -> tuple[float, float]:
            return ((PROBE_OS * sign, 0.0) if axis == "h"
                    else (0.0, PROBE_OS * sign))

        # ---- run the two probes ----
        p0 = measure()
        if p0 is None:
            print("[cal] could not see orange dot at start of probe")
            return None

        d1x, d1y = probe_vec(axis1, sign1)
        move_cursor_rel(d1x, d1y)
        time.sleep(SETTLE)
        p1 = measure()
        move_cursor_rel(-d1x, -d1y)   # return to ~start
        time.sleep(SETTLE)
        if p1 is None:
            print(f"[cal] lost orange dot after probe1 ({axis1})")
            return None

        p_mid = measure()
        if p_mid is None:
            p_mid = p0  # fallback

        d2x, d2y = probe_vec(axis2, sign2)
        move_cursor_rel(d2x, d2y)
        time.sleep(SETTLE)
        p2 = measure()
        move_cursor_rel(-d2x, -d2y)
        time.sleep(SETTLE)
        if p2 is None:
            print(f"[cal] lost orange dot after probe2 ({axis2})")
            return None

        # Columns: panel response per OS unit along the SCREEN axes.
        # probe1 sent (d1x, d1y) OS px and produced (p1-p0) panel px.
        # That measures column `axis1` of J after dividing by sign*PROBE_OS.
        col1 = ((p1[0] - p0[0]) / (PROBE_OS * sign1),
                (p1[1] - p0[1]) / (PROBE_OS * sign1))
        col2 = ((p2[0] - p_mid[0]) / (PROBE_OS * sign2),
                (p2[1] - p_mid[1]) / (PROBE_OS * sign2))
        # Place columns in the (h, v) order expected by callers:
        # J[:,0] = response per OS-X, J[:,1] = response per OS-Y.
        col_h = col1 if axis1 == "h" else col2
        col_v = col2 if axis1 == "h" else col1
        J = np.array([[col_h[0], col_v[0]],
                      [col_h[1], col_v[1]]], dtype=np.float64)

        # Sanity: each column needs nontrivial magnitude (the game
        # actually responded). If a column is near zero, calibration is
        # garbage -- the game ate the input or the dot didn't move.
        if (np.linalg.norm(col_h) < 0.05) or (np.linalg.norm(col_v) < 0.05):
            print(f"[cal] probe response too small: J=\n{J}")
            return None
        if abs(np.linalg.det(J)) < 1e-4:
            print(f"[cal] near-singular J:\n{J}")
            return None
        return J

    def walk_path(self, plan: PlanResult, debug_dir: Path | None = None):
        """Calibrate-then-drive walker.

        Step 1 (once per trigger): send two small mouse probes and measure
        the resulting orange-dot motion in a canonical panel coordinate
        system. Each calibration sample re-detects the panel in a fresh
        frame and re-warps it into that canonical size, so calibration
        still works even if the panel moves after trigger time.

        Step 2 (per tick): capture a fresh frame, re-detect the panel,
        re-warp it into that same canonical size, locate the current cell
        and orange dot there, then convert panel error to an OS mouse
        delta via ``J^-1``. Magnitude is capped at ``--step``. As soon as
        the cyan ring reaches the target cell we advance to the next path
        step.
        """
        path = plan.path
        trace_points: list[tuple[float, float, bool]] = []
        end_reason = "unknown"

        try:
            if not path or len(path) < 2:
                end_reason = "nothing_to_walk"
                print("[walk] nothing to walk")
                return

            path_set = {cell: i for i, cell in enumerate(path)}
            H = plan.panel_bgr.shape[0]
            W = plan.panel_bgr.shape[1]

            # ---- 1. calibrate ----
            print("[walk] calibrating mouse->panel mapping...")
            J = self._calibrate_jacobian(W, H, plan)
            if J is None:
                end_reason = "calibration_failed"
                print("[walk] calibration failed; aborting walk")
                return
            J_inv = np.linalg.inv(J)
            print(f"[walk] J (panel_px / OS_px):\n"
                  f"       [[{J[0,0]:+.3f} {J[0,1]:+.3f}]\n"
                  f"        [{J[1,0]:+.3f} {J[1,1]:+.3f}]]")

            # ---- 2. drive ----
            STEP_MAX = max(40.0, self.step_px)
            TICK = self.tick_s
            t_start = time.time()
            last_cell_logged = path[0]
            last_target = None
            ticks_in_hop = 0

            while True:
                if self.abort_walk:
                    end_reason = "aborted_esc"
                    print("[walk] aborted (ESC)")
                    return
                if not (self.rmb_down and self.x1_down):
                    end_reason = "trigger_released"
                    print("[walk] aborted (trigger keys released)")
                    return

                ticks_in_hop += 1
                if ticks_in_hop > WALK_MAX_TICKS_PER_HOP:
                    end_reason = "hop_budget_exceeded"
                    print("[walk] hop budget exceeded; giving up")
                    return

                panel_view = self._capture_panel_view(W, H)
                if panel_view is None:
                    if TICK > 0:
                        time.sleep(TICK)
                    continue
                panel = panel_view.panel_bgr

                cur_cell = find_current_cell(panel, plan.n_rows, plan.n_cols,
                                             exclude=plan.goal)
                if cur_cell is None:
                    if TICK > 0:
                        time.sleep(TICK)
                    continue

                # Panel-pixel position of the dot. Prefer cell-gated detector
                # (more robust against red icons), fall back to whole-panel.
                orange = find_orange_centroid_in_cell(
                    panel, cur_cell, plan.n_rows, plan.n_cols)
                orange_direct = orange is not None
                if orange is None:
                    p_any = detect_orange_current(panel)
                    orange = (p_any["center"][0], p_any["center"][1]) if p_any else None
                if orange is None:
                    cur_px = (cur_cell[1] + 0.5) * plan.cell_w
                    cur_py = (cur_cell[0] + 0.5) * plan.cell_h
                else:
                    cur_px, cur_py = orange
                trace_points.append((float(cur_px), float(cur_py), orange_direct))

                # ---- pick target cell (advance whenever cyan ring reaches it) ----
                on_path = cur_cell in path_set
                if on_path:
                    cur_idx = path_set[cur_cell]
                    if cur_idx >= len(path) - 1:
                        end_reason = "reached_goal"
                        print(f"[walk] reached goal in "
                              f"{time.time() - t_start:.2f}s")
                        return
                    target = path[cur_idx + 1]
                else:
                    target = min(path, key=lambda p: abs(p[0] - cur_cell[0])
                                 + abs(p[1] - cur_cell[1]))

                if cur_cell != last_cell_logged:
                    last_cell_logged = cur_cell
                    tag = "on-path" if on_path else "OFF-path -> recovering"
                    print(f"[walk] now at {cur_cell} ({tag}, target {target})")

                if target != last_target:
                    ticks_in_hop = 0
                    last_target = target

                tgt_px = (target[1] + 0.5) * plan.cell_w
                tgt_py = (target[0] + 0.5) * plan.cell_h
                err_panel = np.array([tgt_px - cur_px, tgt_py - cur_py])

                # Convert panel-pixel error to OS-pixel mouse delta.
                # Damping (<1) absorbs small calibration error and prevents
                # the overshoot/correct/overshoot oscillation that looked
                # like "infinite flipping".
                os_delta = WALK_DAMPING * (J_inv @ err_panel)
                mag = float(np.linalg.norm(os_delta))
                if mag < 1.0:
                    # Already at target in OS terms; let the cyan ring catch up.
                    if TICK > 0:
                        time.sleep(TICK)
                    continue
                if mag > STEP_MAX:
                    os_delta = os_delta * (STEP_MAX / mag)
                    mag = STEP_MAX

                # MOVE -> SETTLE -> (next iter captures fresh frame).
                move_cursor_rel(float(os_delta[0]), float(os_delta[1]))
                if self.debug:
                    print(f"[walk] err=({err_panel[0]:+.0f},{err_panel[1]:+.0f}) px "
                          f"-> os=({os_delta[0]:+.0f},{os_delta[1]:+.0f}) "
                          f"mag={mag:.0f}")

                if TICK > 0:
                    time.sleep(TICK)
        finally:
            if debug_dir is not None:
                self._write_walk_trace_debug(debug_dir, plan, trace_points,
                                             end_reason)

    # ---- main loop ----
    def run_console(self, hwnd: int, title: str):
        self.target_hwnd = hwnd
        self.target_title = title
        self.engine = Engine()
        self.start_listeners()
        print(f"\n[ready] target window: {title!r} (hwnd={hwnd})")
        print(f"[walker] step={self.step_px:.0f} px, tick={self.tick_s:.3f} s")
        if self.debug:
            print(f"[debug] ON \u2192 artifacts will be saved under {DEBUG_DIR}")
        print("Hold RIGHT MOUSE + MOUSE 4, then tap T to trigger.")
        print("ESC or releasing RMB/M4 aborts the active walk. Ctrl+C to quit.")
        try:
            while True:
                try:
                    ev = self.event_q.get(timeout=0.25)
                except Empty:
                    continue
                if ev == "TRIGGER":
                    self.trigger()
        except KeyboardInterrupt:
            print("\nbye.")


# --- Tk window picker --------------------------------------------------------

def pick_window_gui() -> tuple[int, str, bool] | None:
    import tkinter as tk
    from tkinter import ttk

    wins = list_visible_windows()
    chosen = {"hwnd": None, "title": "", "debug": False}

    root = tk.Tk()
    root.title("Hack-Panel Assist 鈥?select game window")
    root.geometry("700x460")

    ttk.Label(root, text="Pick the game window:").pack(anchor="w", padx=12, pady=8)
    frame = ttk.Frame(root)
    frame.pack(fill="both", expand=True, padx=12)
    sb = ttk.Scrollbar(frame, orient="vertical")
    lb = tk.Listbox(frame, yscrollcommand=sb.set, font=("Consolas", 10))
    sb.config(command=lb.yview)
    sb.pack(side="right", fill="y")
    lb.pack(side="left", fill="both", expand=True)
    for hwnd, title in wins:
        lb.insert("end", f"{hwnd:>10}  {title}")

    def refresh():
        lb.delete(0, "end")
        wins.clear()
        wins.extend(list_visible_windows())
        for hwnd, title in wins:
            lb.insert("end", f"{hwnd:>10}  {title}")

    debug_var = tk.BooleanVar(value=False)

    def confirm():
        sel = lb.curselection()
        if not sel:
            return
        hwnd, title = wins[sel[0]]
        chosen["hwnd"] = hwnd
        chosen["title"] = title
        chosen["debug"] = bool(debug_var.get())
        root.destroy()

    opts = ttk.Frame(root)
    opts.pack(fill="x", padx=12, pady=(8, 0))
    ttk.Checkbutton(
        opts,
        text="Debug mode (dump capture, bbox, warp, cells, overlay per trigger)",
        variable=debug_var,
    ).pack(anchor="w")

    bar = ttk.Frame(root)
    bar.pack(fill="x", pady=8, padx=12)
    ttk.Button(bar, text="Refresh", command=refresh).pack(side="left")
    ttk.Button(bar, text="Start", command=confirm).pack(side="right")
    lb.bind("<Double-Button-1>", lambda e: confirm())

    root.mainloop()
    if chosen["hwnd"] is None:
        return None
    return chosen["hwnd"], chosen["title"], chosen["debug"]


# --- entry -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hwnd", type=int, default=None,
                    help="Skip the window picker; use this raw HWND.")
    ap.add_argument("--list", action="store_true",
                    help="List visible windows and exit.")
    ap.add_argument("--step", type=float, default=WALK_STEP_PX,
                    help=f"OS-pixel magnitude per mouse nudge (default {WALK_STEP_PX}).")
    ap.add_argument("--tick", type=float, default=WALK_TICK_S,
                    help=f"Sleep seconds between ticks (default {WALK_TICK_S}).")
    ap.add_argument("--debug", action="store_true",
                    help="Dump per-trigger artifacts under panel_detector/runs/debug/.")
    args = ap.parse_args()

    if args.list:
        for hwnd, title in list_visible_windows():
            print(f"{hwnd:>12}  {title}")
        return

    if args.hwnd is not None:
        hwnd = args.hwnd
        title = win32gui.GetWindowText(hwnd)
        debug = args.debug
    else:
        picked = pick_window_gui()
        if picked is None:
            print("no window chosen, exiting")
            return
        hwnd, title, debug = picked
        debug = debug or args.debug

    AssistApp(step_px=args.step, tick_s=args.tick, debug=debug).run_console(hwnd, title)


if __name__ == "__main__":
    main()
