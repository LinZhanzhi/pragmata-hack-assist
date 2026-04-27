// UI controller: generates inputs, renders the grid, runs the solver,
// and visualizes the result path.

const $ = (id) => document.getElementById(id);

let state = {
  grid: [],
  start: null,
  dest: null,
  path: null,
};

function generateGrid() {
  const rows = parseInt($("rows").value, 10);
  const cols = parseInt($("cols").value, 10);
  const rewardPct = parseInt($("rewardPct").value, 10) / 100;
  const forbiddenPct = parseInt($("forbiddenPct").value, 10) / 100;

  const grid = Array.from({ length: rows }, () => new Array(cols).fill(CELL.NORMAL));

  // Random start and destination (distinct)
  const start = [randInt(rows), randInt(cols)];
  let dest;
  do {
    dest = [randInt(rows), randInt(cols)];
  } while (dest[0] === start[0] && dest[1] === start[1]);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if ((r === start[0] && c === start[1]) || (r === dest[0] && c === dest[1])) continue;
      const v = Math.random();
      if (v < forbiddenPct) grid[r][c] = CELL.FORBIDDEN;
      else if (v < forbiddenPct + rewardPct) grid[r][c] = CELL.REWARD;
    }
  }
  grid[start[0]][start[1]] = CELL.START;
  grid[dest[0]][dest[1]] = CELL.DEST;

  state = { grid, start, dest, path: null };
  renderGrid();
  setStatus("generated");
  $("time").textContent = "-";
  $("steps").textContent = "-";
  $("rewards").textContent = "-";
  $("expansions").textContent = "-";
}

function randInt(n) { return Math.floor(Math.random() * n); }

const CELL_SIZE = 22;
const CELL_GAP = 1;

function renderGrid() {
  const { grid } = state;
  const rows = grid.length, cols = grid[0].length;
  const gridEl = $("grid");
  gridEl.style.gridTemplateColumns = `repeat(${cols}, ${CELL_SIZE}px)`;
  gridEl.innerHTML = "";

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cell = document.createElement("div");
      cell.className = "cell";
      const v = grid[r][c];
      if (v === CELL.START) cell.classList.add("start");
      else if (v === CELL.DEST) cell.classList.add("dest");
      else if (v === CELL.REWARD) cell.classList.add("reward");
      else if (v === CELL.FORBIDDEN) cell.classList.add("forbidden");
      cell.title = `(${r},${c})`;
      gridEl.appendChild(cell);
    }
  }

  drawPathOverlay();
}

function drawPathOverlay() {
  const svg = $("pathOverlay");
  const gridEl = $("grid");
  const w = gridEl.offsetWidth;
  const h = gridEl.offsetHeight;
  svg.setAttribute("width", w);
  svg.setAttribute("height", h);
  svg.innerHTML = "";

  const path = state.path;
  if (!path || path.length < 2) return;

  // Cell center: border(1) + col*(size+gap) + size/2
  const center = (idx) => 1 + idx * (CELL_SIZE + CELL_GAP) + CELL_SIZE / 2;
  const points = path.map(([r, c]) => `${center(c)},${center(r)}`).join(" ");

  const poly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  poly.setAttribute("points", points);
  poly.setAttribute("fill", "none");
  poly.setAttribute("stroke", "#ff3b30");
  poly.setAttribute("stroke-width", "3");
  poly.setAttribute("stroke-linecap", "round");
  poly.setAttribute("stroke-linejoin", "round");
  svg.appendChild(poly);
}

function solveCurrent() {
  if (!state.grid.length) {
    setStatus("generate a grid first");
    return;
  }
  const alpha = parseFloat($("alpha").value);
  const beta = parseFloat($("beta").value);
  const maxExpansions = parseInt($("maxExp").value, 10);

  setStatus("solving...");

  // Defer to next frame so the UI can update.
  requestAnimationFrame(() => {
    const t0 = performance.now();
    const result = solve(state.grid, state.start, state.dest, {
      alpha, beta, maxExpansions,
    });
    const t1 = performance.now();

    state.path = result.path;
    renderGrid();

    if (result.path) {
      setStatus("done");
      $("steps").textContent = result.path.length - 1;
      $("rewards").textContent = result.rewards;
    } else {
      setStatus("no path found");
      $("steps").textContent = "-";
      $("rewards").textContent = "-";
    }
    $("time").textContent = (t1 - t0).toFixed(2);
    $("expansions").textContent = result.expansions;
  });
}

function clearPath() {
  state.path = null;
  renderGrid();
  setStatus("path cleared");
}

function setStatus(s) { $("status").textContent = s; }

$("generateBtn").addEventListener("click", generateGrid);
$("solveBtn").addEventListener("click", solveCurrent);
$("clearBtn").addEventListener("click", clearPath);

generateGrid();
