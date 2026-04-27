import glob, os, numpy as np
W,H=1920,1080
TOP=-0.0839
BOT=0.0326
files=sorted(glob.glob(r"d:\pathSolver\panel_detector\raw\*.txt"))
errs={k:[] for k in ["TL","TR","BR","BL"]}
rows=[]
for f in files:
    with open(f) as fh:
        line=fh.readline().strip().split()
    vals=list(map(float,line[1:]))
    cx,cy,w,h=vals[0:4]
    kpts=vals[4:]
    gt=[]
    for i in range(4):
        gt.append((kpts[i*3]*W, kpts[i*3+1]*H))
    bxl=(cx-w/2)*W; bxr=(cx+w/2)*W
    byt=(cy-h/2)*H; byb=(cy+h/2)*H
    pred={
        "TL":(bxl, byt+TOP*(bxl-bxr)),
        "TR":(bxr, byt),
        "BR":(bxr, byb),
        "BL":(bxl, byb+BOT*(bxl-bxr)),
    }
    names=["TL","TR","BR","BL"]
    for i,n in enumerate(names):
        dx=pred[n][0]-gt[i][0]; dy=pred[n][1]-gt[i][1]
        errs[n].append((dx*dx+dy*dy)**0.5)
    rows.append((bxl,bxr,byt,byb,gt))

print(f"Files: {len(files)}")
print("Fixed slopes TOP=%.4f BOT=%.4f"%(TOP,BOT))
all_e=[]
for n in ["TL","TR","BR","BL"]:
    a=np.array(errs[n]); all_e.extend(a.tolist())
    print(f"  {n}: mean={a.mean():.2f}px max={a.max():.2f}px")
print(f"  Overall mean: {np.mean(all_e):.2f}px")

# Refit: TL.y_pred = byt + s_top*(bxl-bxr); residual = gt_TL.y - byt over (bxl-bxr)
num_t=den_t=num_b=den_b=0.0
for (bxl,bxr,byt,byb,gt) in rows:
    dx=bxl-bxr
    num_t += dx*(gt[0][1]-byt)
    den_t += dx*dx
    num_b += dx*(gt[3][1]-byb)
    den_b += dx*dx
TOP2=num_t/den_t; BOT2=num_b/den_b
print(f"\nFitted slopes TOP={TOP2:.5f} BOT={BOT2:.5f}")

errs2={k:[] for k in ["TL","TR","BR","BL"]}
for (bxl,bxr,byt,byb,gt) in rows:
    pred={
        "TL":(bxl, byt+TOP2*(bxl-bxr)),
        "TR":(bxr, byt),
        "BR":(bxr, byb),
        "BL":(bxl, byb+BOT2*(bxl-bxr)),
    }
    for i,n in enumerate(["TL","TR","BR","BL"]):
        dx=pred[n][0]-gt[i][0]; dy=pred[n][1]-gt[i][1]
        errs2[n].append((dx*dx+dy*dy)**0.5)
all_e=[]
for n in ["TL","TR","BR","BL"]:
    a=np.array(errs2[n]); all_e.extend(a.tolist())
    print(f"  {n}: mean={a.mean():.2f}px max={a.max():.2f}px")
print(f"  Overall mean: {np.mean(all_e):.2f}px")
