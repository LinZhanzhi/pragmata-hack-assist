import os, glob, statistics
D=r"d:\pathSolver\panel_detector\raw"
W,H=1920,1080
rows=[]
tops=[]; bots=[]; lefts=[]; rights=[]
tops_n=[]; bots_n=[]
trdx=[]; trdy=[]; brdx=[]; brdy=[]
files=sorted(glob.glob(os.path.join(D,"*.txt")))
print(f"Found {len(files)} files")
hdr=("file","top_px","bot_px","left_px","right_px","TRdx","TRdy","BRdx","BRdy")
print("{:<30} {:>9} {:>9} {:>9} {:>9} {:>8} {:>8} {:>8} {:>8}".format(*hdr))
for f in files:
    with open(f) as fh:
        parts=fh.read().split()
    if len(parts)<17: continue
    vals=list(map(float,parts))
    cls=vals[0]; cx,cy,w,h=vals[1:5]
    kp=[(vals[5+i*3],vals[6+i*3]) for i in range(4)]
    TL,TR,BR,BL=kp
    def slope(a,b,scale=False):
        ax,ay=a; bx,by=b
        if scale:
            ax*=W; ay*=H; bx*=W; by*=H
        dx=bx-ax
        if abs(dx)<1e-9: return float('inf')
        return (by-ay)/dx
    top_n=slope(TL,TR); bot_n=slope(BL,BR)
    top_p=slope(TL,TR,True); bot_p=slope(BL,BR,True)
    left_p=slope(TL,BL,True); right_p=slope(TR,BR,True)
    bx_r=cx+w/2; by_t=cy-h/2; by_b=cy+h/2
    tr_dx=(TR[0]-bx_r)*W; tr_dy=(TR[1]-by_t)*H
    br_dx=(BR[0]-bx_r)*W; br_dy=(BR[1]-by_b)*H
    tops.append(top_p); bots.append(bot_p); lefts.append(left_p); rights.append(right_p)
    tops_n.append(top_n); bots_n.append(bot_n)
    trdx.append(tr_dx); trdy.append(tr_dy); brdx.append(br_dx); brdy.append(br_dy)
    name=os.path.basename(f)[:28]
    print(f"{name:<30} {top_p:>9.4f} {bot_p:>9.4f} {left_p:>9.2f} {right_p:>9.2f} {tr_dx:>8.2f} {tr_dy:>8.2f} {br_dx:>8.2f} {br_dy:>8.2f}")

def ms(a):
    return (statistics.mean(a), statistics.pstdev(a))
print()
print("=== Summary (pixel space, 1920x1080) ===")
for name,a in [("top_slope_px",tops),("bot_slope_px",bots),("left_slope_px",[x for x in lefts if x!=float('inf')]),("right_slope_px",[x for x in rights if x!=float('inf')])]:
    m,s=ms(a); print(f"{name}: mean={m:+.5f} std={s:.5f} min={min(a):+.4f} max={max(a):+.4f}")
print()
print("=== Summary (normalized) ===")
for name,a in [("top_slope_n",tops_n),("bot_slope_n",bots_n)]:
    m,s=ms(a); print(f"{name}: mean={m:+.5f} std={s:.5f}")
print()
print("=== TR/BR vs bbox corners (pixels) ===")
for name,a in [("TR_dx",trdx),("TR_dy",trdy),("BR_dx",brdx),("BR_dy",brdy)]:
    m,s=ms(a); print(f"{name}: mean={m:+.3f} std={s:.3f} min={min(a):+.3f} max={max(a):+.3f}")
