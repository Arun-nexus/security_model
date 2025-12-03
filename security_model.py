import cv2
import numpy as np
import time

try:
    import winsound
    wb = True
except:
    wb = False

cap = cv2.VideoCapture(0)
rf = 0.6
rects = []
dflag = False
ex = ey = ix = iy = -1

def cb(e,x,y,f,p):
    global dflag, ex, ey, ix, iy, fr, rects
    if e == 1:
        dflag = True
        ex,ey = x,y
    elif e == 0 and dflag:
        tmp = fr.copy()
        cv2.rectangle(tmp,(ex,ey),(x,y),(0,0,255),1)
        cv2.imshow("win",tmp)
    elif e == 4:
        dflag = False
        ix,iy = x,y
        rects.append(((ex,ey),(ix,iy)))

cv2.namedWindow("win")
cv2.setMouseCallback("win",cb)

lh = np.array([0,30,60],dtype=np.uint8)
uh = np.array([25,255,255],dtype=np.uint8)

pt = time.time()
fps = 0

def lg(msk,ma=1000):
    cs,_ = cv2.findContours(msk,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs,key=cv2.contourArea)
    if cv2.contourArea(c)<ma:
        return None
    return c

def fp(cnt):
    t = tuple(cnt[cnt[:,:,1].argmin()][0])
    m = cv2.moments(cnt)
    if m["m00"]!=0:
        cx = int(m["m10"]/m["m00"])
        cy = int(m["m01"]/m["m00"])
    else:
        cx,cy = t
    return (cx,cy),t

def pd(p,r):
    (x1,y1),(x2,y2)=r
    x1,x2=min(x1,x2),max(x1,x2)
    y1,y2=min(y1,y2),max(y1,y2)
    x,y=p
    if x1<=x<=x2 and y1<=y<=y2:
        return 0
    dx = max(x1-x,0,x-x2)
    dy = max(y1-y,0,y-y2)
    return np.hypot(dx,dy)

while True:
    ret,fo = cap.read()
    if not ret:
        break

    fr = cv2.resize(fo,(0,0),fx=rf,fy=rf)
    vs = fr.copy()

    hsv = cv2.cvtColor(fr,cv2.COLOR_BGR2HSV)
    msk = cv2.inRange(hsv,lh,uh)

    if msk is None or msk.size == 0:
        cv2.imshow("win",vs)
        if cv2.waitKey(1)&0xFF==ord('x'):
            break
        continue

    if msk.dtype != np.uint8:
        msk = msk.astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))

    msk = cv2.morphologyEx(msk,cv2.MORPH_OPEN,k)
    msk = cv2.morphologyEx(msk,cv2.MORPH_CLOSE,k)
    msk = cv2.GaussianBlur(msk,(7,7),0)

    cn = lg(msk,1000)
    fpnt = None
    cpt = None

    if cn is not None:
        cv2.drawContours(vs,[cn],-1,(0,255,0),2)
        h = cv2.convexHull(cn)
        cv2.drawContours(vs,[h],-1,(0,128,255),2)
        cpt,fpnt = fp(cn)
        fpnt = (int(fpnt[0]),int(fpnt[1]))
        cpt = (int(cpt[0]),int(cpt[1]))
        cv2.circle(vs,fpnt,6,(0,0,255),-1)
        cv2.circle(vs,cpt,5,(255,0,0),-1)

    for i,r in enumerate(rects):
        (x1,y1),(x2,y2)=r
        x1,x2=min(x1,x2),max(x1,x2)
        y1,y2=min(y1,y2),max(y1,y2)
        cv2.rectangle(vs,(x1,y1),(x2,y2),(0,0,255),2)

        dg = np.hypot(x2-x1,y2-y1)+1e-6
        dt = 0.12*dg
        wt = 0.30*dg

        st = "no hand"
        col = (200,200,200)

        if fpnt is not None:
            d = pd(fpnt,((x1,y1),(x2,y2)))
            if d==0 or d<=dt:
                st="danger"
                col=(0,0,255)
            elif d<=wt:
                st="warning"
                col=(0,165,255)
            else:
                st="safe"
                col=(0,255,0)

        cv2.putText(vs,f"r{i}: {st}",(x1,max(15,y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)

        if st=="danger":
            h,w=vs.shape[:2]
            cv2.putText(vs,"danger danger",(int(w*0.15),int(h*0.5)),
                        cv2.FONT_HERSHEY_DUPLEX,2,(0,0,255),4)
            if wb:
                try:
                    winsound.Beep(1000,150)
                except:
                    pass

    nt = time.time()
    fps = 0.9*fps + 0.1*(1/(nt-pt+1e-6))
    pt = nt
    cv2.putText(vs,f"fps: {fps:.1f}",(10,20),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

    if len(rects)==0:
        cv2.putText(vs,"draw rectangele or  press x to exit",
                    (10, vs.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)

    cv2.imshow("win",vs)

    k=cv2.waitKey(1)&0xFF
    if k==ord('c'):
        rects=[]
    if k==ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
