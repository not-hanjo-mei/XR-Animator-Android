import json, numpy as np, math
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
P=json.load(open('calib/persons.json'))
def crop(im, cx, cy, size, ang, out):
    # rotated square crop centred at (cx,cy), side=size, rotation ang (rad), resized to out
    c,s=math.cos(ang),math.sin(ang)
    # affine: output pixel (u,v) -> input
    k=size/out
    a=(k*c, -k*s, cx - k*(c*out/2 - s*out/2), k*s, k*c, cy - k*(s*out/2 + c*out/2))
    return np.asarray(im.transform((out,out), Image.AFFINE, a, resample=Image.BILINEAR), np.float32)/255.
pose,hand,face=[],[],[]
for p in P:
    im=Image.open('calib/img/'+p['file']).convert('RGB')
    x,y,w,h=p['bbox']; kp=np.array(p['kp'],np.float32).reshape(17,3)
    s=max(w,h)*1.25; pose.append(crop(im,x+w/2,y+h/2,s,0,256))
    # face: nose 0, eyes 1,2, ears 3,4
    v=kp[:5,2]>0
    if v[0] and v[1] and v[2]:
        pts=kp[:5][v,:2]; eyes=kp[1:3,:2]
        d=np.linalg.norm(eyes[0]-eyes[1]); ear=np.ptp(pts[:,0]) if v[3] or v[4] else 0
        sz=max(d*4.0, ear*1.8)
        ang=math.atan2(eyes[1,1]-eyes[0,1], eyes[1,0]-eyes[0,0])
        if ang>math.pi/2: ang-=math.pi
        if ang<-math.pi/2: ang+=math.pi
        c=(eyes.mean(0)*0.5+kp[0,:2]*0.5)
        if sz>40: face.append(crop(im,c[0],c[1],sz,ang,256))
    # hands: elbow 7/8, wrist 9/10
    for e,wi in ((7,9),(8,10)):
        if kp[e,2]>0 and kp[wi,2]>0:
            fa=kp[wi,:2]-kp[e,:2]; L=np.linalg.norm(fa)
            if L<30: continue
            c=kp[wi,:2]+fa*0.35; ang=math.atan2(fa[1],fa[0])+math.pi/2
            # mediapipe hand crops: wrist at bottom, fingers up -> rotate so forearm points down
            hand.append(crop(im,c[0],c[1],L*1.1,ang,224))
def keep(arr, model, pres_idx, thr):
    it=Interpreter(model_path=model); it.allocate_tensors()
    inp=it.get_input_details()[0]['index']; out=it.get_output_details()
    ok=[]
    for a in arr:
        it.set_tensor(inp,a[None]); it.invoke()
        v=float(np.array(it.get_tensor(out[pres_idx]['index'])).ravel()[0])
        ok.append(v)
    ok=np.array(ok); return ok
for name,arr,model in (('pose',pose,'models/pose_landmarker_full/pose_landmarks_detector.tflite'),('hand',hand,'models/hand_landmarker/hand_landmarks_detector.tflite'),('face',face,'models/face_landmarker/face_landmarks_detector.tflite')):
    arr=np.stack(arr)
    sc=keep(arr,model,1,0)
    print(name,len(arr),'presence pct',np.percentile(sc,[10,50,90]))
    np.save(f'calib/{name}.npy',arr); np.save(f'calib/{name}_score.npy',sc)
