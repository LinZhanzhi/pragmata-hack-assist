"""Render any ready-bucket frame for spot-checking."""
import sys, cv2
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from dest_detector import infer_grid_from_dest, detect_green_dest

ids = sys.argv[1:] or ['s01_0108', 's01_0141', 's01_0143']
out = ROOT / 'runs' / 'dest_gate' / 'spotcheck'
out.mkdir(parents=True, exist_ok=True)
for fid in ids:
    p = ROOT / 'raw' / f'{fid}_panel.png'
    bgr = cv2.imread(str(p))
    if bgr is None:
        print(f'{fid}: NOT FOUND'); continue
    H, W = bgr.shape[:2]
    g = infer_grid_from_dest(bgr)
    d = detect_green_dest(bgr)
    vis = bgr.copy()
    if g:
        cw, ch = g['cell_w'], g['cell_h']
        for r in range(g['n_rows']+1):
            cv2.line(vis,(0,int(r*ch)),(W,int(r*ch)),(0,255,255),1)
        for c in range(g['n_cols']+1):
            cv2.line(vis,(int(c*cw),0),(int(c*cw),H),(0,255,255),1)
        x,y,w,h = g['icon_bbox']
        cv2.rectangle(vis,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(vis,f"{g['n_rows']}x{g['n_cols']} dest=({g['dest_row']},{g['dest_col']})",
                    (5,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
    elif d:
        x,y,w,h = d['bbox']
        cv2.rectangle(vis,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(vis,f"REJECT sol={d['solidity']:.2f} a={d['area']}",
                    (5,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    cv2.imwrite(str(out/f'{fid}.png'), vis)
    print(f'{fid}: {g}')
