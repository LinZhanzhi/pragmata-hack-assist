Given 20×20, 4-dir moves, and “good enough + very fast” instead of optimal, I’d use a greedy / best-first search that balances “go toward goal” and “pick up nearby rewards”, with a hard no-revisit rule.

***

## 1. High-level idea

Use a graph search (like A*) but change the cost so it prefers:

- Staying roughly on a path toward the destination.
- Going through reward cells when they are not too detour-y.
- Never revisiting a cell (so you keep paths simple and short).

You don’t care about optimality, so you can:

- Use a cheap heuristic (Manhattan distance to goal). [theory.stanford](http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html)
- Bias the search to expand nodes with more collected rewards.
- Stop as soon as you reach the goal the first time.

This will run comfortably under milliseconds on 20×20.

***

## 2. Simple weighted A* style approach

Define for a state:

- Position: \((x, y)\)
- Collected reward count: \(r\)
- Path cost so far: \(g\) (e.g. steps taken)
- Heuristic to goal: \(h = |x - x_T| + |y - y_T|\) (Manhattan distance) [diva-portal](https://www.diva-portal.org/smash/get/diva2:832863/FULLTEXT01.pdf)

Then define a “score” to **maximize**:

\[
\text{score}(state) = \alpha \cdot r - \beta \cdot (g + h)
\]

Where:

- \(\alpha > 0\): how much you value each reward.
- \(\beta > 0\): how strongly you prefer short / direct paths.

Algorithm:

1. Use a priority queue ordered by highest score.
2. Start with state at S:
   - \(r = 1\) if S is a reward, else 0.
   - \(g = 0\).
3. Maintain a visited set of cells on the current path only (like DFS), not global, to enforce “no cell twice”.
4. Pop best state, if at goal, return its path immediately.
5. For each 4-neighbor:
   - Skip if out of bounds, forbidden, or already in this path’s visited set.
   - New \(r' = r + 1\) if neighbor is reward and not yet visited in this path.
   - New \(g' = g + 1\).
   - Compute \(h'\), score, push child state.

Because you stop at the first time you reach T, this is more like a greedy best-first search, not full A*. [ravi-bhide.blogspot](http://ravi-bhide.blogspot.com/2011/11/greedy-algorithm-for-grid-navigation.html)
In practice, on a 20×20, branching factor ≤ 4, this is very fast.

You can tune:

- Large \(\alpha\): more greedy for rewards, may wander more.
- Large \(\beta\): more direct, fewer rewards.

***

## 3. Even simpler: “move toward goal but sniff rewards”

If you want something extremely simple and CPU-cheap, you can use a one-step greedy:

At each step:

1. From current cell, consider up to 4 neighbors that are:
   - In bounds.
   - Not forbidden.
   - Not visited yet.
2. For each candidate neighbor, compute:
   - Reward bonus: 1 if it’s a reward cell, else 0.
   - Goal proximity: negative Manhattan distance to goal (closer is better).
3. Score each neighbor:

\[
\text{local\_score} = \alpha \cdot \text{reward\_bonus} - \beta \cdot \text{dist\_to\_goal}
\]

4. Move to neighbor with max local_score.
5. If no neighbors, you’re stuck: either give up or restart with slightly different parameters / random tie-breaking.

This is \(O(\text{path length})\) time, almost constant for your grid size. [algodaily](https://algodaily.com/lessons/getting-to-know-greedy-algorithms-through-examples/greedy-algorithm-for-maximizing-reward)
It’s very easy to implement and very responsive, but more likely to get stuck or miss clusters of rewards behind small detours.

***

## 4. Practical recommendation for your case

Given your constraints and that 20×20 is tiny:

- Implement the “score-based best-first search” (section 2).
  - It’s still cheap enough to run in real time per frame.
  - It will explore more than the purely local greedy and tends to find better paths.
- Add a node expansion limit (e.g. 1–3k states) to ensure worst-case latency is bounded; if limit is hit, fall back to “just go to goal via simple A* ignoring rewards” for safety.

Sketch in code-ish pseudo:

```python
from heapq import heappush, heappop

def find_path(grid, start, goal, reward_set, forbidden_set,
              alpha=5.0, beta=1.0, max_expansions=3000):
    R, C = len(grid), len(grid[0])

    def manhattan(a, b):
        return abs(a[0]-b[0]) + abs(a [theory.stanford](http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html)-b [theory.stanford](http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html))

    # priority queue: (-score, unique_id, state)
    # state: (x, y, g, r, path, visited_set)
    pq = []
    uid = 0

    r0 = 1 if start in reward_set else 0
    g0 = 0
    h0 = manhattan(start, goal)
    score0 = alpha * r0 - beta * (g0 + h0)
    heappush(pq, (-score0, uid, (start[0], start [theory.stanford](http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html), g0, r0, [start], {start})))
    uid += 1

    expansions = 0

    while pq and expansions < max_expansions:
        _, _, (x, y, g, r, path, visited) = heappop(pq)
        expansions += 1

        if (x, y) == goal:
            return path, r

        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if not (0 <= nx < R and 0 <= ny < C):
                continue
            if (nx, ny) in forbidden_set:
                continue
            if (nx, ny) in visited:
                continue

            g2 = g + 1
            r2 = r + (1 if (nx, ny) in reward_set else 0)
            h2 = manhattan((nx, ny), goal)
            score2 = alpha * r2 - beta * (g2 + h2)

            new_path = path + [(nx, ny)]
            new_visited = visited | {(nx, ny)}

            heappush(pq, (-score2, uid, (nx, ny, g2, r2, new_path, new_visited)))
            uid += 1

    # fallback: just go to goal ignoring rewards (A* by distance) or return None
    return None, 0
```

You can tweak:

- `alpha` to control how aggressive you are about rewards.
- `max_expansions` to guarantee responsiveness.

This avoids revisiting cells by construction, avoids forbidden cells, and normally picks up many rewards while still heading toward the goal.

***

Would you like to bias it more toward “never getting trapped” (i.e., always guarantee reaching the goal) or more toward “collecting many rewards even if route is long”?