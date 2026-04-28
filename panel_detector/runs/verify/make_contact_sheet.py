import cv2, numpy as np, os
DIR = r"d:\pathSolver\panel_detector\runs\verify\cells"
OUT = r"d:\pathSolver\panel_detector\runs\verify"

def make(frame, N):
    cell = 64
    border = 2
    sep = 4
    cb = cell + 2*border  # cell with black border
    size = N*cb + (N+1)*sep
    img = np.full((size, size, 3), 255, np.uint8)
    for r in range(N):
        for c in range(N):
            p = os.path.join(DIR, f"{frame}_N{N}_r{r}_c{c}.png")
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is None:
                raise FileNotFoundError(p)
            if im.shape[:2] != (cell, cell):
                im = cv2.resize(im, (cell, cell))
            bordered = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, value=(0,0,0))
            y = sep + r*(cb + sep)
            x = sep + c*(cb + sep)
            img[y:y+cb, x:x+cb] = bordered
    out = os.path.join(OUT, f"contact_{frame}.png")
    cv2.imwrite(out, img)
    print(out)

make("s01_0001", 4)
make("s01_0005", 5)
make("s01_0009", 5)
