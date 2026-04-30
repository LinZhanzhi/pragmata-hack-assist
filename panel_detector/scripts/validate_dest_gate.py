"""Validate dest_detector gate on the s01 warped-panel dataset.

Buckets each frame as:
  ready    - passes fully-loaded gate, plausible grid
  rejected - has a green blob but fails gate (mid-animation, bogus YOLO, ...)
  nodet    - no green blob at all (likely bogus YOLO panel)

Renders rejected/nodet frames to runs/dest_gate/{rejected,nodet}/<fid>.png so
the user can verify the gate isn't kicking out real panels.
"""
from pathlib import Path
import json
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from dest_detector import detect_green_dest, infer_grid_from_dest, _green_mask  # noqa: E402

RAW = ROOT / 'raw'
OUT = ROOT / 'runs' / 'dest_gate'
(OUT / 'rejected').mkdir(parents=True, exist_ok=True)
(OUT / 'nodet').mkdir(parents=True, exist_ok=True)


def render(bgr, info, label):
    H, W = bgr.shape[:2]
    mask = cv2.cvtColor(_green_mask(bgr), cv2.COLOR_GRAY2BGR)
    overlay = bgr.copy()
    if info is not None:
        x, y, w, h = info['bbox']
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)
        txt = (f"a={info['area']} ar={info['aspect']:.2f} "
               f"sol={info['solidity']:.2f} ready={info['ready']}")
        cv2.putText(overlay, txt, (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)
    sheet = np.hstack([bgr, mask, overlay])
    cv2.putText(sheet, label, (5, H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return sheet


def main():
    files = sorted(RAW.glob('s01_*_panel.png'))
    summary = {'ready': [], 'rejected': [], 'nodet': []}
    grid_hist = {}
    results = {}
    for f in files:
        fid = f.stem
        bgr = cv2.imread(str(f))
        if bgr is None:
            continue
        d = detect_green_dest(bgr)
        g = infer_grid_from_dest(bgr) if d is not None else None
        if d is None:
            summary['nodet'].append(fid)
            cv2.imwrite(str(OUT / 'nodet' / f'{fid}.png'),
                        render(bgr, None, f'{fid}  NODET'))
            results[fid] = {'bucket': 'nodet'}
        elif g is None:
            summary['rejected'].append(fid)
            cv2.imwrite(str(OUT / 'rejected' / f'{fid}.png'),
                        render(bgr, d, f'{fid}  REJECTED'))
            results[fid] = {'bucket': 'rejected', **d, 'bbox': list(d['bbox'])}
        else:
            summary['ready'].append(fid)
            key = f"{g['n_rows']}x{g['n_cols']}"
            grid_hist[key] = grid_hist.get(key, 0) + 1
            results[fid] = {'bucket': 'ready',
                            **{k: (list(v) if isinstance(v, tuple) else v)
                               for k, v in g.items()}}

    n = len(files)
    print(f'total={n}')
    print(f'  ready    = {len(summary["ready"]):3d}  ({len(summary["ready"])/n:.1%})')
    print(f'  rejected = {len(summary["rejected"]):3d}  ({len(summary["rejected"])/n:.1%})')
    print(f'  nodet    = {len(summary["nodet"]):3d}  ({len(summary["nodet"])/n:.1%})')
    print('\ninferred grid histogram (ready bucket):')
    for k, c in sorted(grid_hist.items(), key=lambda x: -x[1]):
        print(f'  {k}: {c}')
    if summary['rejected']:
        print('\nrejected frames:')
        for fid in summary['rejected']:
            r = results[fid]
            print(f'  {fid}: bbox={r["bbox"]} '
                  f'aspect={r["aspect"]:.2f} sol={r["solidity"]:.2f} '
                  f'area={r["area"]}')
    if summary['nodet']:
        print('\nnodet frames:', summary['nodet'])

    with open(OUT / 'results.json', 'w') as fp:
        json.dump({'summary': {k: v for k, v in summary.items()},
                   'grid_hist': grid_hist,
                   'frames': results}, fp, indent=2)
    print(f'\nwrote {OUT/"results.json"}')
    print(f'rejected sheets -> {OUT/"rejected"}')
    print(f'nodet sheets    -> {OUT/"nodet"}')


if __name__ == '__main__':
    main()
