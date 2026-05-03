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
from current_detector import find_current_cell  # noqa: E402
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

ALPHA, BETA = 5.0, 1.0
# Closed-loop walker tunables. The game's mouse sensitivity is unknown, so
# we don't try to be clever: pick a starting OS-pixel step, increase if no
# progress, halve on overshoot, give up after a budget per cell.
WALK_TICK_S = 0.04           # delay between capture/move iterations
WALK_STEP_INIT_PX = 25       # initial OS-pixel magnitude per nudge
WALK_STEP_MIN_PX = 4
WALK_STEP_MAX_PX = 200
WALK_NO_PROGRESS_GROW = 6    # iterations w/ no cell change before growing step
WALK_MAX_TICKS_PER_HOP = 200 # ~8s per cell at 40ms tick


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

    def plan(self, client_bgr: np.ndarray) -> PlanResult | None:
        # 1. detect panel
        results = self.yolo.predict(source=client_bgr, conf=0.25, verbose=False)
        if not results or len(results[0].boxes) == 0:
            print("[plan] no panel detected")
            return None
        boxes = results[0].boxes
        best = int(boxes.conf.argmax().item())
        xyxy = boxes.xyxy[best].cpu().numpy()
        kpts = corners_from_bbox(xyxy)  # 4x2 in client coords (TL,TR,BR,BL)

        out_w, out_h = warp_size_from_bbox(xyxy)
        dst = np.array([[0, 0], [out_w - 1, 0],
                        [out_w - 1, out_h - 1], [0, out_h - 1]],
                       dtype=np.float32)
        M = cv2.getPerspectiveTransform(kpts, dst)
        M_inv = np.linalg.inv(M)
        panel = cv2.warpPerspective(client_bgr, M, (out_w, out_h))

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
        for r in range(n_rows):
            for c in range(n_cols):
                cell = panel[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
                rgb = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
                crops.append(self.tf(Image.fromarray(rgb)))
        batch = torch.stack(crops).to(self.device)
        with torch.no_grad():
            probs = F.softmax(self.cls_model(batch), dim=1).cpu().numpy()
        labels = np.empty((n_rows, n_cols), dtype=object)
        for i in range(n_rows * n_cols):
            r, c = divmod(i, n_cols)
            labels[r, c] = self.classes[int(probs[i].argmax())]

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
    def __init__(self):
        self.engine: Engine | None = None
        self.target_hwnd: int | None = None
        self.target_title: str = ""
        self.rmb_down = False
        self.x1_down = False
        self.armed = True
        self.busy = False
        self.abort_walk = False
        self.event_q: Queue = Queue()

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
            plan = self.engine.plan(client)
            if plan is None:
                return
            cv2.imwrite(str(LIVE_DIR / f"{ts}_overlay.png"), plan.overlay_bgr)
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

        Re-detects the current cyan ring every tick and nudges the mouse
        in the screen-space direction toward the next path cell. Adapts
        step magnitude based on whether the detected cell is changing.
        Aborts on ESC, on RMB or M4 release, on a hop budget overrun, or
        when the goal is reached.
        """
        path = plan.path
        if not path or len(path) < 2:
            print("[walk] nothing to walk")
            return

        # Precompute screen-space vectors for each (a -> b) hop. The relative
        # mouse delta needed in OS pixels is unknown (sensitivity), but the
        # *direction* on screen is well-defined by the panel homography.
        def panel_to_client(pcx, pcy):
            v = plan.M_inv @ np.array([pcx, pcy, 1.0])
            return v[0] / v[2], v[1] / v[2]

        # Current target index (in `path`); we are moving toward path[idx].
        idx = 1
        step_px = float(WALK_STEP_INIT_PX)
        last_cell = path[0]
        no_progress = 0
        ticks_in_hop = 0
        t_start = time.time()

        while idx < len(path):
            # ---- abort conditions ----
            if self.abort_walk:
                print("[walk] aborted (ESC)")
                return
            if not (self.rmb_down and self.x1_down):
                print("[walk] aborted (trigger keys released)")
                return

            target = path[idx]
            ticks_in_hop += 1
            if ticks_in_hop > WALK_MAX_TICKS_PER_HOP:
                print(f"[walk] hop budget exceeded at target {target}; giving up")
                return

            # ---- capture + detect ----
            client = capture_window_client(self.target_hwnd)
            if client is None:
                time.sleep(WALK_TICK_S)
                continue
            panel = cv2.warpPerspective(
                client, np.linalg.inv(plan.M_inv),
                (plan.panel_bgr.shape[1], plan.panel_bgr.shape[0]))
            cur = find_current_cell(panel, plan.n_rows, plan.n_cols,
                                    exclude=plan.goal)

            # ---- progress logic ----
            if cur is not None and cur != last_cell:
                no_progress = 0
                last_cell = cur
                # Snap step back down on every successful cell transition;
                # this prevents runaway acceleration once we're in motion.
                step_px = max(WALK_STEP_MIN_PX, step_px * 0.5)
                print(f"[walk] now at {cur} (target {target}, idx {idx}/"
                      f"{len(path) - 1})")

            # If the detected cell jumped *past* the immediate target, fast
            # forward through any path cells we already crossed.
            if cur is not None:
                while idx < len(path) and cur == path[idx]:
                    idx += 1
                    ticks_in_hop = 0
                    no_progress = 0
                    if idx >= len(path):
                        break
                if idx >= len(path):
                    break
                target = path[idx]
                # Also fast-forward if cur appears later in the path (we
                # overshot one or more cells).
                for j in range(idx + 1, len(path)):
                    if cur == path[j]:
                        idx = j + 1
                        ticks_in_hop = 0
                        no_progress = 0
                        break
                if idx >= len(path):
                    break
                target = path[idx]

            # ---- compute screen-direction toward target ----
            cur_for_dir = cur if cur is not None else last_cell
            ar, ac = cur_for_dir
            br, bc = target
            # Centers in panel pixel space.
            apx = (ac + 0.5) * plan.cell_w
            apy = (ar + 0.5) * plan.cell_h
            bpx = (bc + 0.5) * plan.cell_w
            bpy = (br + 0.5) * plan.cell_h
            ax, ay = panel_to_client(apx, apy)
            bx, by = panel_to_client(bpx, bpy)
            vx, vy = bx - ax, by - ay
            length = (vx * vx + vy * vy) ** 0.5
            if length < 1e-3:
                time.sleep(WALK_TICK_S)
                continue
            vx /= length
            vy /= length

            # ---- adaptive step ----
            no_progress += 1
            if no_progress >= WALK_NO_PROGRESS_GROW:
                step_px = min(WALK_STEP_MAX_PX, step_px * 1.5)
                no_progress = 0
                print(f"[walk] no progress, step -> {step_px:.0f} px")

            # If we have a lock on cur != target, also dampen perpendicular
            # error: only nudge along the cell-grid axis with the larger
            # delta to avoid diagonal drift inside a cell.
            if abs(br - ar) > abs(bc - ac):
                vx = 0.0
                vy = 1.0 if br > ar else -1.0
            else:
                vy = 0.0
                vx = 1.0 if bc > ac else -1.0

            move_cursor_rel(vx * step_px, vy * step_px)
            time.sleep(WALK_TICK_S)

        print(f"[walk] reached goal in {time.time() - t_start:.2f}s")

    # ---- main loop ----
    def run_console(self, hwnd: int, title: str):
        self.target_hwnd = hwnd
        self.target_title = title
        self.engine = Engine()
        self.start_listeners()
        print(f"\n[ready] target window: {title!r} (hwnd={hwnd})")
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
    args = ap.parse_args()

    if args.list:
        for hwnd, title in list_visible_windows():
            print(f"{hwnd:>12}  {title}")
        return

    if args.hwnd is not None:
        hwnd = args.hwnd
        title = win32gui.GetWindowText(hwnd)
    else:
        picked = pick_window_gui()
        if picked is None:
            print("no window chosen, exiting")
            return
        hwnd, title = picked

    AssistApp().run_console(hwnd, title)


if __name__ == "__main__":
    main()
