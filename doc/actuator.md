# Actuator — Plan

> Status: planning only. The cell classifier is not trained yet (more
> raw frames are needed to cover all board sizes and tile types). This
> document captures the design of the **actuator** — the component that
> takes a solved path and drives the player's mouse to walk the cursor
> dot through the panel cells.

## Game interaction model (what the player does)

Capture context: keyboard + mouse (no controller).

- **Aim**: hold the right mouse button (`RMB`).
  - With nothing hackable in view, no panel appears.
  - Some hackable objects show a *different* puzzle, not the path
    panel. The system must detect "is this a path panel" before doing
    anything.
- **Engage path puzzle**: while still holding `RMB`, also hold
  `Left Alt`.
  - A **yellow / orange dot** appears in the panel.
  - The dot represents the current path head, originating at the start
    cell.
  - Mouse motion now moves the *dot inside the panel* (no longer
    free-aim).
  - Releasing `LAlt` (or `RMB`) cancels / drops the path.

This means the actuator only ever runs while both `RMB` and `LAlt` are
held. Outside that combination it must be inert.

### Implications

1. **Default behavior is "do nothing"**. The tool runs continuously but
   only acts when the panel is visible *and* we're in path-engagement
   mode.
2. **Start cell is visually obscured** by the yellow dot. We cannot
   rely on a `power_active` icon at the start cell. In practice the
   start cell tends to be a plain walkable cell anyway, so the
   classifier should be allowed to predict `empty` there. The
   *position of the start* is recovered from **dot tracking**, not from
   tile classification.
3. **Path execution is mouse-driven**. The cursor dot moves in
   response to relative mouse motion, not absolute coordinates. We
   need a closed-loop controller: track the dot position frame to
   frame, compare to the next path waypoint, push a small mouse delta,
   repeat.

## Architecture

```text
[screen capture loop]
   |
   v
[panel detector + warp + grid-N + cell classifier]
   |   (only when RMB+LAlt held)
   v
[dot tracker]  --- yields (row, col) of cursor in panel coords
   |
   v
[solver]  --- given grid + start (dot) + destination, produces path
   |
   v
[actuator]  --- drives mouse_move deltas to walk the dot along path
   |
   v
[supervisor]  --- monitors held keys, abort/reset, safety timeouts
```

### Modules

- **input_monitor**: low-level hook for `RMB` and `LAlt` state. On
  Windows, `GetAsyncKeyState` polled at the capture rate is fine.
- **screen_grab**: capture the game window region. Use `mss` or DXGI
  desktop duplication. Target ≥ 30 fps.
- **panel_pipeline**: existing
  `predict.py` -> `infer_grid.py` -> `slice_cells.py` -> classifier.
  Reused as a function.
- **dot_tracker**: see §Dot tracking.
- **solver**: see §Solver interface.
- **actuator_driver**: see §Mouse driving.
- **supervisor**: state machine that wires the above and enforces
  safety: hard abort on key release, max-step budget, watchdog if the
  dot stops moving despite mouse output.

## Dot tracking

The yellow/orange dot is a small saturated blob with a distinctive hue
(yellow ≈ HSV (20–35, S>180, V>200)) that doesn't appear elsewhere on
the panel by default. Plan:

1. On the warped panel image (the same 800×800 we already produce),
   threshold in HSV for the dot color.
2. Take the largest connected component; its centroid is the dot
   position in panel coordinates.
3. Map centroid `(x, y)` to grid cell `(row, col)` using the gridline
   positions returned by `infer_grid` (we already compute those).
4. Maintain a short history (last 5 frames) for velocity / stability:
   - the dot is "settled" in cell `(r, c)` when it has stayed in that
     cell for ≥ 2 frames.
5. If the threshold yields no blob:
   - either the path puzzle is not engaged (no dot rendered yet),
   - or the dot is occluded by a tile icon. Treat as "unknown
     position", do not actuate, wait.

Calibration: capture a few path-engaged frames once we have them, find
the actual dot color empirically (it may be more orange than yellow,
and may shift slightly when on a reward tile). Make the HSV ranges a
config constant tuned from real samples, not guessed.

### Why dot tracking and not "just count steps"

Mouse-to-dot mapping is non-linear: panel scale changes with board
dimension, in-game sensitivity may be variable, the dot can stick on a
tile briefly, etc. Open-loop "send N pixels right per cell" will
desynchronize quickly. Closed-loop tracking is the only reliable way.

## Solver interface

The solver is a separate module (already partially implemented in
`solver.js` for the web prototype; will be re-implemented or wrapped
in Python for the runtime). For the actuator it must expose:

```python
def solve(grid: list[list[Role]],
          start: tuple[int, int],
          destination: tuple[int, int] | None = None
          ) -> list[tuple[int, int]]:
    """Return a sequence of cells (start, ..., destination) to walk
    through, attempting to maximize rewards collected and avoid
    forbidden cells. Cells are 4-neighbor connected. The first cell
    in the returned list is `start`.
    """
```

Notes:

- Destination is sometimes implicit (e.g. "reach any `power_idle`
  tile"); the solver must accept either an explicit target or a
  predicate.
- The solver is allowed to return `None` if no path exists. The
  supervisor handles that gracefully (do nothing, surface a message).

## Mouse driving

Two layers:

1. **Direction step**: from current cell `(r, c)` to next path cell
   `(r', c')`, the required move is one of `up / down / left / right`.
   Translate to a mouse delta `(dx, dy)` in pixels. The magnitude is
   tuned by an online controller (see below); the sign is given by
   the direction.

2. **Closed-loop micro-controller** per cell transition:
   ```
   target = path[i + 1]
   while not dot_settled_at(target) and not aborted:
       observed = current_dot_cell()
       if observed == target: break
       delta = compute_delta(observed_subpixel_pos, target_center)
       mouse_move_relative(delta.x, delta.y)
       sleep(frame_period)
   ```
   `compute_delta` is a P-controller in panel pixel space scaled by an
   estimated panel-pixels-per-mouse-count gain. The gain is bootstrapped
   from a tiny calibration push at engage time and refined online.

3. **Mouse API**: on Windows use `SendInput` with
   `MOUSEEVENTF_MOVE | MOUSEEVENTF_VIRTUALDESK` and `dx, dy` deltas, or
   the `interception` driver if `SendInput` is filtered by anti-cheat.
   Start with `SendInput`; only escalate if needed.

### Pacing

- Cap mouse delta per tick to a safe maximum so we never overshoot more
  than ~half a cell.
- Throttle to the screen-grab rate so we always make decisions on fresh
  observations.
- Add a small randomized jitter on the timing so it doesn't look
  perfectly mechanical (optional, if anti-cheat heuristics worry us).

## Trigger / toggle design

You raised the right question. Two viable modes:

### Mode A — "passive auto-pilot": user holds RMB+LAlt, tool runs

- The tool is always on; it watches `RMB` and `LAlt`.
- When both are held *and* a path panel is detected *and* the dot is
  visible, the actuator engages and walks the path.
- When either key releases, abort immediately.

Pros:
- Zero extra UI for the user; matches normal in-game muscle memory.
- Safe-by-default — releasing a key always cancels.

Cons:
- The user can never solve the puzzle manually while the tool is
  running (tool will fight them for the mouse).
- A momentary stutter / wrong solve mid-engagement could feel
  inescapable.

### Mode B — explicit toggle (recommended)

- Bind a global hotkey (e.g. `Ctrl+Shift+H` for "hack assist") that
  toggles the tool **armed / disarmed**.
- When **disarmed**: tool only observes (for debugging / data
  capture); never moves the mouse.
- When **armed**: behaves like Mode A — engage on RMB+LAlt, abort on
  release.
- An on-screen indicator (small overlay or system tray icon color)
  shows the armed state.

Pros:
- User stays in control. They can leave the tool disarmed when they
  want to play normally, and arm it only when they want help.
- Doubles as a kill switch.

Cons:
- One extra hotkey to remember.

### Recommendation

Build **Mode B**. The toggle is cheap to add and the safety upside is
large. Defaults:

| Hotkey            | Action                                      |
|-------------------|---------------------------------------------|
| `Ctrl+Shift+H`    | Toggle armed / disarmed.                    |
| `Ctrl+Shift+P`    | Pause/resume capture loop entirely.         |
| `Esc` (while engaged) | Hard abort current solve, leave armed. |

Add a third "soft" safety: if the dot doesn't move for `N` consecutive
ticks despite us pushing nonzero mouse deltas, abort and disarm. This
catches "user grabbed the mouse" or "game lost focus" cases.

## State machine (supervisor)

```
DISARMED  --(toggle hotkey)-->  ARMED_IDLE
ARMED_IDLE  --(RMB held)-->  ARMED_AIMING
ARMED_AIMING  --(LAlt held + panel + dot)-->  ENGAGED
ENGAGED    --(solve OK)-->  EXECUTING
EXECUTING  --(reach dest OR abort OR key release)-->  ARMED_IDLE
ARMED_*    --(toggle hotkey)-->  DISARMED
any        --(panic hotkey)-->  DISARMED
```

Transitions out of `EXECUTING` always release any synthetic input.

## Safety / anti-cheat considerations

- **Never click**, only move. Walking the dot only requires
  `MOUSEEVENTF_MOVE`, not buttons. Less likely to look like an
  aimbot to heuristics.
- **Cap motion per tick**. No teleport-fast deltas. Keep dot velocity
  inside the band a human could produce.
- **Disarm by default on launch**. No accidental engagement.
- **Disarm on focus loss**. If the game window isn't foreground, no
  motion.
- **Visible indicator**. Always show armed status so the user knows.
- The ethics / TOS question is the user's to weigh. From a
  build-perspective we just make the tool predictable and disarmable.

## Calibration

One-time per game session, on first engagement:

1. Push a tiny test mouse delta (e.g. `+2, 0`) and measure how many
   panel pixels the dot moved.
2. Repeat for the orthogonal axis.
3. Store gain `(panel_px_per_mouse_count_x, _y)`.

If the user changes in-game sensitivity mid-session, the controller's
online refinement (see §Mouse driving) absorbs that gradually; the
calibration just gets us to a sensible starting point.

## Open questions (resolve before implementing)

1. **Dot color** — confirm exact HSV range from real captures. Does it
   change on reward / forbidden cells? On overlap with an icon?
2. **Dot lag** — is there visible latency between mouse motion and dot
   update? If yes, the controller needs anti-windup so it doesn't
   over-push during the lag window.
3. **Path completion signal** — how does the game tell us the puzzle
   succeeded? A panel disappearance? A flash? We need a clear
   end-of-puzzle signal so the supervisor can return to `ARMED_IDLE`.
4. **Failure modes** — can the path be invalidated mid-execution
   (e.g. a forbidden tile lights up)? If yes, the supervisor needs to
   re-classify periodically and re-plan.
5. **Multiple panels** — do we ever see more than one panel at a
   time? If yes, pick the largest / centermost.
6. **Other puzzle types** — the user mentioned non-path puzzles can
   appear. The classifier upstream must include an "this is not a
   path panel" outcome (or we add a separate "panel type"
   classifier). Treat unrecognized panels as "do nothing".

## What to build first (MVP order)

Once the cell classifier is trained:

1. **Read-only loop**: capture → detect panel → infer grid → classify
   cells → track dot → print state to stdout. No mouse output.
   Verify it stays inert outside `RMB+LAlt`.
2. **Toggle + hotkeys**: the supervisor state machine, no actuation
   yet.
3. **Calibration probe**: one tiny mouse push, log the resulting dot
   delta. Verify the closed-loop tracker sees it.
4. **Single-step actuator**: walk one cell only (start → adjacent
   cell), then disengage. Iterate on gain / pacing.
5. **Full path execution**: chain single steps with the supervisor and
   abort logic.
6. **End-of-puzzle detection**: figure out the success signal and
   return to idle cleanly.

Each MVP step should land behind a runtime flag so we can advance
without breaking the previous one.
