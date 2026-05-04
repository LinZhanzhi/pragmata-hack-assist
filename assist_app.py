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
WALK_TICK_S = 0.0            # delay between capture/move iterations (0 = no sleep, fastest)
WALK_STEP_PX = 75            # default OS-pixel magnitude per nudge (overridable via --step)
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

    def plan(self, client_bgr: np.ndarray,
             debug_dir: "Path | None" = None) -> "PlanResult | None":
        # 1. detect panel
        results = self.yolo.predict(source=client_bgr, conf=0.25, verbose=False)
        if not results or len(results[0].boxes) == 0:
            print("[plan] no panel detected")
            if debug_dir is not None:
                cv2.imwrite(str(debug_dir / "01_capture.png"), client_bgr)
            return None
        boxes = results[0].boxes
        best = int(boxes.conf.argmax().item())
        xyxy = boxes.xyxy[best].cpu().numpy()
        kpts = corners_from_bbox(xyxy)  # 4x2 in client coords (TL,TR,BR,BL)

        if debug_dir is not None:
            cv2.imwrite(str(debug_dir / "01_capture.png"), client_bgr)
            vis = client_bgr.copy()
            x0, y0, x1, y1 = (int(round(v)) for v in xyxy[:4])
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            corner_colors = [(0, 0, 255), (0, 255, 255),
                             (255, 0, 255), (255, 255, 0)]  # TL TR BR BL
            for (cx, cy), col in zip(kpts, corner_colors):
                cv2.circle(vis, (int(round(cx)), int(round(cy))), 6, col, -1)
            cv2.putText(vis, f"conf={float(boxes.conf[best]):.2f}",
                        (x0, max(0, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
            cv2.imwrite(str(debug_dir / "02_capture_with_bbox.png"), vis)

        out_w, out_h = warp_size_from_bbox(xyxy)
        dst = np.array([[0, 0], [out_w - 1, 0],
                        [out_w - 1, out_h - 1], [0, out_h - 1]],
                       dtype=np.float32)
        M = cv2.getPerspectiveTransform(kpts, dst)
        M_inv = np.linalg.inv(M)
        panel = cv2.warpPerspective(client_bgr, M, (out_w, out_h))

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
            self.walk_path(plan)
        except Exception:
            traceback.print_exc()
        finally:
            self.busy = False

    def walk_path(self, plan: PlanResult):
        """Closed-loop walker.

        Per tick we re-capture and warp the game window, find the cyan-ring
        cell (= which cell the cursor is in) and the orange-dot centroid
        inside that cell (= precise sub-cell position). Then:
          - if current cell is on the planned path, target = next path cell
            after it; if it's the last one, we're done.
          - if current cell is OFF-path (overshoot or drift), target = the
            nearest cell on the path so we recover before continuing.
        Direction: we compare the orange dot's panel-pixel position to the
        target cell's center in panel space, pick the axis with the larger
        remaining delta, map that single panel axis to a screen axis via
        M_inv (sign of dominant component), and nudge a fixed 50 px in
        that one screen axis. Only horizontal OR vertical mouse motion --
        never diagonal.
        Aborts on ESC, on RMB or M4 release, on a hop budget overrun.
        """
        path = plan.path
        if not path or len(path) < 2:
            print("[walk] nothing to walk")
            return

        path_set = {cell: i for i, cell in enumerate(path)}
        H = plan.panel_bgr.shape[0]
        W = plan.panel_bgr.shape[1]
        M_panel_from_client = np.linalg.inv(plan.M_inv)

        def panel_axis_to_screen_axis(dxp: float, dyp: float) -> tuple[float, float]:
            """Map a panel-space direction (dxp, dyp) to a single screen
            axis (sx, sy) where exactly one of sx/sy is +/-1 and the other
            is 0. We sample two panel points 100 px apart along the chosen
            panel axis, project both to client/screen via M_inv, then snap
            the resulting screen vector to its dominant component.
            """
            origin_p = np.array([W * 0.5, H * 0.5, 1.0])
            tip_p = np.array([W * 0.5 + dxp * 100.0,
                              H * 0.5 + dyp * 100.0, 1.0])
            o = plan.M_inv @ origin_p
            t = plan.M_inv @ tip_p
            o /= o[2]
            t /= t[2]
            sx_raw = float(t[0] - o[0])
            sy_raw = float(t[1] - o[1])
            if abs(sx_raw) >= abs(sy_raw):
                return (1.0 if sx_raw > 0 else -1.0, 0.0)
            return (0.0, 1.0 if sy_raw > 0 else -1.0)

        last_cell = path[0]
        ticks_in_hop = 0
        last_target = None
        t_start = time.time()
        # step_px = float(WALK_STEP_INIT_PX)   # adaptive variant (disabled)
        # no_progress = 0

        while True:
            # ---- abort conditions ----
            if self.abort_walk:
                print("[walk] aborted (ESC)")
                return
            if not (self.rmb_down and self.x1_down):
                print("[walk] aborted (trigger keys released)")
                return

            ticks_in_hop += 1
            if ticks_in_hop > WALK_MAX_TICKS_PER_HOP:
                print(f"[walk] hop budget exceeded; giving up")
                return

            # ---- capture + warp to panel ----
            client = capture_window_client(self.target_hwnd)
            if client is None:
                if self.tick_s > 0:
                    time.sleep(self.tick_s)
                continue
            panel = cv2.warpPerspective(client, M_panel_from_client, (W, H))

            # ---- detect current cell + orange dot ----
            cur_cell = find_current_cell(panel, plan.n_rows, plan.n_cols,
                                         exclude=plan.goal)
            if cur_cell is None:
                # No cyan ring detected -- can't safely move. Skip this tick.
                if self.tick_s > 0:
                    time.sleep(self.tick_s)
                continue

            orange = find_orange_centroid_in_cell(
                panel, cur_cell, plan.n_rows, plan.n_cols)
            if orange is not None:
                cur_px, cur_py = orange
            else:
                cur_px = (cur_cell[1] + 0.5) * plan.cell_w
                cur_py = (cur_cell[0] + 0.5) * plan.cell_h

            # ---- pick target cell ----
            on_path = cur_cell in path_set
            if on_path:
                cur_idx = path_set[cur_cell]
                if cur_idx >= len(path) - 1:
                    print(f"[walk] reached goal in "
                          f"{time.time() - t_start:.2f}s")
                    return
                target = path[cur_idx + 1]
            else:
                # nearest path cell (Manhattan)
                target = min(path, key=lambda p: abs(p[0] - cur_cell[0])
                             + abs(p[1] - cur_cell[1]))

            if cur_cell != last_cell:
                # step_px = max(WALK_STEP_MIN_PX, step_px * 0.5)  # adaptive
                # no_progress = 0
                last_cell = cur_cell
                tag = "on-path" if on_path else "OFF-path -> recovering"
                print(f"[walk] now at {cur_cell} ({tag}, target {target})")

            if target != last_target:
                ticks_in_hop = 0
                last_target = target

            # ---- panel-space direction toward target center ----
            tgt_px = (target[1] + 0.5) * plan.cell_w
            tgt_py = (target[0] + 0.5) * plan.cell_h
            dx_p = tgt_px - cur_px
            dy_p = tgt_py - cur_py

            # tolerance: if we're inside the target cell already, the
            # cyan-ring re-detection on the next tick will advance us.
            if abs(dx_p) < 1.0 and abs(dy_p) < 1.0:
                if self.tick_s > 0:
                    time.sleep(self.tick_s)
                continue

            # only one panel axis at a time -> one screen axis at a time
            if abs(dx_p) >= abs(dy_p):
                sx, sy = panel_axis_to_screen_axis(1.0 if dx_p > 0 else -1.0, 0.0)
            else:
                sx, sy = panel_axis_to_screen_axis(0.0, 1.0 if dy_p > 0 else -1.0)

            # adaptive step (disabled):
            # no_progress += 1
            # if no_progress >= WALK_NO_PROGRESS_GROW:
            #     step_px = min(WALK_STEP_MAX_PX, step_px * 1.5)
            #     no_progress = 0
            #     print(f"[walk] no progress, step -> {step_px:.0f} px")

            move_cursor_rel(sx * self.step_px, sy * self.step_px)
            if self.tick_s > 0:
                time.sleep(self.tick_s)

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

def pick_window_gui() -> tuple[int, str] | None:
    import tkinter as tk
    from tkinter import ttk

    wins = list_visible_windows()
    chosen = {"hwnd": None, "title": ""}

    root = tk.Tk()
    root.title("Hack-Panel Assist – select game window")
    root.geometry("700x420")

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

    def confirm():
        sel = lb.curselection()
        if not sel:
            return
        hwnd, title = wins[sel[0]]
        chosen["hwnd"] = hwnd
        chosen["title"] = title
        root.destroy()

    bar = ttk.Frame(root)
    bar.pack(fill="x", pady=8, padx=12)
    ttk.Button(bar, text="Refresh", command=refresh).pack(side="left")
    ttk.Button(bar, text="Start", command=confirm).pack(side="right")
    lb.bind("<Double-Button-1>", lambda e: confirm())

    root.mainloop()
    if chosen["hwnd"] is None:
        return None
    return chosen["hwnd"], chosen["title"]


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
                    help="Sleep seconds between ticks; 0 = fastest (default 0).")
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
