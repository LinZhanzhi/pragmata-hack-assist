// Cell type constants
const CELL = {
  NORMAL: 0,
  FORBIDDEN: 1,
  REWARD: 2,
  START: 3,
  DEST: 4,
};

/**
 * Min-heap priority queue.
 * Items are compared by their `.priority` (lower = higher priority).
 */
class MinHeap {
  constructor() { this.h = []; }
  size() { return this.h.length; }
  push(item) {
    const h = this.h;
    h.push(item);
    let i = h.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (h[p].priority <= h[i].priority) break;
      [h[p], h[i]] = [h[i], h[p]];
      i = p;
    }
  }
  pop() {
    const h = this.h;
    if (h.length === 0) return undefined;
    const top = h[0];
    const last = h.pop();
    if (h.length > 0) {
      h[0] = last;
      let i = 0;
      const n = h.length;
      while (true) {
        const l = i * 2 + 1, r = i * 2 + 2;
        let best = i;
        if (l < n && h[l].priority < h[best].priority) best = l;
        if (r < n && h[r].priority < h[best].priority) best = r;
        if (best === i) break;
        [h[best], h[i]] = [h[i], h[best]];
        i = best;
      }
    }
    return top;
  }
}

function manhattan(ax, ay, bx, by) {
  return Math.abs(ax - bx) + Math.abs(ay - by);
}

/**
 * Reward-aware best-first search.
 * Maximizes:  score = alpha * rewards - beta * (g + h)
 * The priority queue is a min-heap on (-score), so highest score pops first.
 *
 * To bound state space, we keep the best (lowest g, highest r) per cell:
 * a new visit to a cell is only expanded if it has more rewards OR fewer steps
 * than any previously seen visit to that cell.
 *
 * @param {number[][]} grid
 * @param {[number,number]} start [row, col]
 * @param {[number,number]} goal  [row, col]
 * @param {object} opts { alpha, beta, maxExpansions }
 * @returns {{path: [number,number][]|null, rewards: number, expansions: number}}
 */
function solve(grid, start, goal, opts = {}) {
  const alpha = opts.alpha ?? 5;
  const beta = opts.beta ?? 1;
  const maxExpansions = opts.maxExpansions ?? 5000;

  const R = grid.length;
  const C = grid[0].length;

  const isReward = (r, c) => grid[r][c] === CELL.REWARD;
  const isBlocked = (r, c) => grid[r][c] === CELL.FORBIDDEN;

  // Pre-collect all reward positions for fast scanning.
  const rewardPositions = [];
  for (let r = 0; r < R; r++)
    for (let c = 0; c < C; c++)
      if (grid[r][c] === CELL.REWARD) rewardPositions.push([r, c]);
  const totalRewards = rewardPositions.length;

  // Encode (r,c) as a single int for cheap Set membership.
  const key = (r, c) => r * C + c;

  // Reward-aware heuristic.  While unvisited rewards remain, "h" is the
  // distance to the *nearest unvisited reward* PLUS the distance from
  // that reward on to the goal -- so the search is pulled toward
  // rewards, not directly toward the goal.  When no unvisited rewards
  // remain (or none are reachable in spirit), it falls back to plain
  // Manhattan-to-goal so the path actually finishes.
  const heuristic = (r, c, visited) => {
    let bestD = Infinity;
    let bestPos = null;
    for (let i = 0; i < rewardPositions.length; i++) {
      const [rr, rc] = rewardPositions[i];
      if (visited.has(key(rr, rc))) continue;
      const d = manhattan(r, c, rr, rc);
      if (d < bestD) { bestD = d; bestPos = rewardPositions[i]; }
    }
    if (bestPos === null) return manhattan(r, c, goal[0], goal[1]);
    return bestD + manhattan(bestPos[0], bestPos[1], goal[0], goal[1]);
  };

  // Heap order: lower priority pops first.  Higher rewards collected
  // and lower h (closer to a reward, then to goal) -> pops first.
  const orderPriority = (rew, h) => -(alpha * rew - beta * h);

  const pq = new MinHeap();

  const r0 = isReward(start[0], start[1]) ? 1 : 0;
  const startVisited = new Set([key(start[0], start[1])]);
  const h0 = heuristic(start[0], start[1], startVisited);
  pq.push({
    priority: orderPriority(r0, h0),
    r: start[0], c: start[1],
    g: 0, rew: r0,
    visited: startVisited,
    parent: null,
  });

  const dirs = [[1,0],[-1,0],[0,1],[0,-1]];
  let expansions = 0;
  let goalNode = null;
  let bestGoalScore = -Infinity;

  while (pq.size() > 0 && expansions < maxExpansions) {
    const node = pq.pop();
    expansions++;

    if (node.r === goal[0] && node.c === goal[1]) {
      const finalScore = alpha * node.rew - beta * node.g;
      if (finalScore > bestGoalScore) {
        bestGoalScore = finalScore;
        goalNode = node;
      }
      continue; // keep searching for a better goal path
    }

    for (const [dr, dc] of dirs) {
      const nr = node.r + dr;
      const nc = node.c + dc;
      if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
      if (isBlocked(nr, nc)) continue;

      const k = key(nr, nc);
      if (node.visited.has(k)) continue; // simple paths only

      const ng = node.g + 1;
      const nrew = node.rew + (isReward(nr, nc) ? 1 : 0);

      // Branch-and-bound: if even collecting *every* remaining reward
      // and walking the minimum remaining distance can't beat the best
      // goal we've already found, drop this branch.
      const remainingRewards = totalRewards - nrew;
      const optimisticH = manhattan(nr, nc, goal[0], goal[1]);
      const optimisticScore =
        alpha * (nrew + remainingRewards) - beta * (ng + optimisticH);
      if (optimisticScore <= bestGoalScore) continue;

      const newVisited = new Set(node.visited);
      newVisited.add(k);

      const nh = heuristic(nr, nc, newVisited);
      pq.push({
        priority: orderPriority(nrew, nh),
        r: nr, c: nc,
        g: ng, rew: nrew,
        visited: newVisited,
        parent: node,
      });
    }
  }

  if (!goalNode) {
    return fallbackAStar(grid, start, goal, expansions);
  }

  const path = [];
  let cur = goalNode;
  while (cur) {
    path.push([cur.r, cur.c]);
    cur = cur.parent;
  }
  path.reverse();

  return { path, rewards: goalNode.rew, expansions };
}

/** Plain A* on Manhattan distance, ignoring rewards. */
function fallbackAStar(grid, start, goal, prevExpansions) {
  const R = grid.length, C = grid[0].length;
  const isBlocked = (r, c) => grid[r][c] === CELL.FORBIDDEN;
  const isReward = (r, c) => grid[r][c] === CELL.REWARD;

  const gScore = Array.from({ length: R }, () => new Array(C).fill(Infinity));
  const parent = Array.from({ length: R }, () => new Array(C).fill(null));
  gScore[start[0]][start[1]] = 0;

  const pq = new MinHeap();
  pq.push({
    priority: manhattan(start[0], start[1], goal[0], goal[1]),
    r: start[0], c: start[1], g: 0,
  });

  const dirs = [[1,0],[-1,0],[0,1],[0,-1]];
  let expansions = 0;
  let found = false;

  while (pq.size() > 0) {
    const node = pq.pop();
    expansions++;
    if (node.g > gScore[node.r][node.c]) continue;
    if (node.r === goal[0] && node.c === goal[1]) { found = true; break; }
    for (const [dr, dc] of dirs) {
      const nr = node.r + dr, nc = node.c + dc;
      if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
      if (isBlocked(nr, nc)) continue;
      const ng = node.g + 1;
      if (ng < gScore[nr][nc]) {
        gScore[nr][nc] = ng;
        parent[nr][nc] = [node.r, node.c];
        pq.push({
          priority: ng + manhattan(nr, nc, goal[0], goal[1]),
          r: nr, c: nc, g: ng,
        });
      }
    }
  }

  if (!found) return { path: null, rewards: 0, expansions: prevExpansions + expansions };

  const path = [];
  let cur = [goal[0], goal[1]];
  while (cur) {
    path.push(cur);
    cur = parent[cur[0]][cur[1]];
  }
  path.reverse();
  let rewards = 0;
  for (const [r, c] of path) if (isReward(r, c)) rewards++;
  return { path, rewards, expansions: prevExpansions + expansions };
}
