# PRAGMATA Hacking Assist Project Background

## Overview

This project focuses on automating support for the hacking mechanic in **PRAGMATA**. The in-game hacking sequence can be modeled as a path puzzle on a 2D grid: starting from a known cell, reaching a target cell, avoiding forbidden cells, and collecting as many reward cells as practical along the route. A higher-reward path increases combat value by applying more damage and stronger debuffs to the enemy during the encounter.

The current solver component is already implemented and operational. It produces a valid, fast, good-enough path rather than attempting an expensive optimal search, which matches the real-time requirements of gameplay.

## Why This Project Exists

The hacking puzzle creates a short decision window inside combat. A human player must quickly read the board, identify start and destination cells, notice reward and forbidden blocks, compute a useful route, and then execute that route before the combat opportunity passes.

This project exists to reduce that reaction burden by splitting the problem into three coordinated subsystems:

1. **Sensor** – read the current game image and convert the puzzle board into structured solver input.
2. **Solver** – compute a responsive, high-value path under the puzzle constraints.
3. **Actuator** – execute the resulting path in the game through input automation.

## Current Status

The solver is considered complete enough for the current stage of development.

Its role is to:

- Accept a board representation derived from the game state.
- Identify start, destination, reward, empty, and forbidden cells.
- Produce a no-repeat path from start to destination.
- Favor reward collection while remaining fast enough for real-time use.

Because responsiveness matters more than provable optimality, the solver is treated as a practical heuristic component rather than a mathematically exact optimizer.

## Next Major Work: Sensor

The next subsystem to build is the **sensor**, whose purpose is to translate raw game visuals into a machine-readable board state.

### Sensor Responsibilities

The sensor should:

- Capture the relevant game region that contains the hacking grid.
- Detect when the hacking UI is active.
- Segment the grid into cells.
- Classify each cell as start, destination, reward, forbidden, or normal.
- Output a clean board model for the solver.

### Sensor Input and Output

**Input:**
- Screenshot frame, cropped game capture, or live video frame.

**Output:**
- Grid dimensions.
- Cell-by-cell classification.
- Coordinates for start and destination.
- Coordinates of reward cells.
- Coordinates of forbidden cells.

### Suggested Sensor Pipeline

A practical pipeline can be built in stages:

1. **UI detection** – determine whether the hacking puzzle is currently visible.
2. **Region localization** – isolate the puzzle panel from the full game image.
3. **Grid extraction** – estimate row/column boundaries.
4. **Cell classification** – classify each block by template matching, color/shape heuristics, OCR, or a trained detector.
5. **Board normalization** – convert the recognition result into the exact input format expected by the solver.

### Sensor Design Notes

For a first version, a deterministic computer vision approach is likely the fastest path:

- Fixed screen region or anchor-based localization.
- Template matching for known tile types.
- Simple confidence thresholds and fallback rules.
- Optional manual calibration for different resolutions or HUD layouts.

If the UI varies too much across scenes, later versions can introduce a learned vision model, but a handcrafted pipeline is usually easier to debug at the prototype stage.

## Next Major Work: Actuator

The other subsystem to build is the **actuator**, whose purpose is to convert the solver output into real game inputs.

### Actuator Responsibilities

The actuator should:

- Receive the ordered path from the solver.
- Map path steps into movement or directional commands.
- Send those commands with timing that the game accepts reliably.
- Stop cleanly if the board changes, the puzzle ends, or recognition confidence drops.

### Actuator Input and Output

**Input:**
- Ordered path such as a sequence of cells or directions.

**Output:**
- Simulated keyboard/controller input that reproduces the path in game.

### Suggested Actuation Pipeline

1. **Path translation** – convert adjacent cells into directions such as up, down, left, right.
2. **Input scheduling** – assign delays, press duration, and release timing.
3. **Execution** – send the input sequence to the game process or the active game window.
4. **Verification** – optionally read back game state after each step to confirm progress.
5. **Abort / recovery** – stop execution if state drift is detected.

### Actuator Design Notes

Important engineering concerns include:

- Consistent timing under variable frame rate.
- Window focus and input routing.
- Recovery if a move is dropped or the puzzle state changes.
- Safety controls such as pause, stop hotkey, and execution timeout.

A robust version should treat execution as a closed loop rather than fire-and-forget. If the sensor can confirm the cursor or current tile after each move, the actuator can correct small mistakes before the full path fails.

## System Architecture

At a high level, the project architecture is:

```text
Game Frame
   -> Sensor
   -> Board Representation
   -> Path Solver
   -> Action Sequence
   -> Actuator
   -> Game Input
```

This separation is useful because each module can be developed, tested, and replaced independently.

- The **sensor** owns perception.
- The **solver** owns decision-making.
- The **actuator** owns execution.

## Engineering Priorities

The project should optimize for the following order of priorities:

1. **Correctness of board reading** – wrong recognition makes every later step fail.
2. **Execution reliability** – the game must receive the intended path consistently.
3. **Latency** – the full pipeline must finish within the combat time window.
4. **Path quality** – better reward collection improves combat effect, but only after the first three priorities are stable.

## Development Roadmap

A sensible implementation roadmap is:

### Phase 1
- Freeze the solver input/output format.
- Build screenshot capture and grid-region cropping.
- Manually label several examples of each block type.

### Phase 2
- Implement baseline cell recognition with templates or heuristics.
- Convert recognized boards into solver input.
- Add replay logs for debugging misclassifications.

### Phase 3
- Implement actuator command playback.
- Tune timings for stable in-game execution.
- Add emergency stop and manual override.

### Phase 4
- Close the loop between sensor and actuator.
- Re-check board state during execution.
- Add confidence scoring and automatic abort when state mismatch is detected.

### Phase 5
- Improve recognition accuracy across resolutions and visual effects.
- Improve solver heuristics if better reward collection is needed.
- Package the whole feature as a single game-assist workflow.

## Risks and Unknowns

The main project risks are:

- Visual ambiguity between tile types.
- UI shifts due to resolution, scaling, effects, or camera changes.
- Input timing instability.
- Desynchronization between recognized state and actual puzzle state.
- Real-time pressure making slow perception or execution unusable.

These risks reinforce an important project principle: reliability and observability matter more than algorithmic elegance.

## Background Summary

This project has already completed its path-solving core. The next work is not better path search, but building the perception layer that reads the hacking board and the execution layer that applies the computed route inside the game.

In short:

- **Solver** is done enough to support the prototype.
- **Sensor** is the next perception problem.
- **Actuator** is the next automation problem.
- The final value of the project comes from integrating all three into a fast, reliable combat assist pipeline.
