import os
import json, numpy as np, sys, math
from PIL import Image
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from pipe import *
A=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../') + '/app/src/main/assets/models/'
P=json.load(open('../calib/persons.json'))
seen=set(); files=[]
for p in P:
    x,y,w,h=p['bbox']
    if p['file'] in seen: continue
    if h>250: seen.add(p['file']); files.append(p['file'])
files=files[:40]
mpP=vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=A+'pose_landmarker_full.task'),num_poses=1))
mpH=vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=A+'hand_landmarker.task'),num_hands=2))
mpF=vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=A+'face_landmarker.task'),num_faces=1,output_face_blendshapes=True))
det=PoseDet(); pl=PoseLmk(); hl=HandLmk(); fl=FaceLmk()
ep=[];ep2=[];eh=[];ef=[];ebs=[];nd=0;nh=[0,0];nf=[0,0]
for f in files:
    im=np.asarray(Image.open('../calib/img/'+f).convert('RGB'))
    H,W=im.shape[:2]; img=im.astype(np.float32)/255
    r=mpP.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=im))
    if not r.pose_landmarks: continue
    ref=np.array([[l.x,l.y] for l in r.pose_landmarks[0]])
    rect,sc=det(img)
    if rect is None: nd+=1; continue
    o=pl(img,rect)
    if o is None: nd+=1; continue
    lm,wl,nxt,_=o
    o2=pl(img,nxt); lm2=o2[0] if o2 else lm
    scale=np.ptp(ref[:,1])*H
    ep.append(np.mean(np.hypot((lm[:,0]-ref[:,0])*W,(lm[:,1]-ref[:,1])*H))/scale)
    ep2.append(np.mean(np.hypot((lm2[:,0]-ref[:,0])*W,(lm2[:,1]-ref[:,1])*H))/scale)
    # hands
    rh=mpH.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=im))
    refs=[np.array([[l.x,l.y] for l in hh]) for hh in rh.hand_landmarks]
    for side in (0,1):
        if lm2[15+side,3]<0.5: continue
        hr=hand_rect_from_pose(lm2,W,H,side); oh=hl(img,hr)
        if oh is None: continue
        oh2=hl(img,oh[3]) or oh
        hp=oh2[0]
        for rf in refs:
            d=np.mean(np.hypot((hp[:,0]-rf[:,0])*W,(hp[:,1]-rf[:,1])*H)); s=np.ptp(rf[:,1])*H+1
            if d/s<0.5: eh.append(d/s); nh[0]+=1
        nh[1]+=1
    rf=mpF.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=im))
    if rf.face_landmarks:
        nf[1]+=1
        reff=np.array([[l.x,l.y] for l in rf.face_landmarks[0]]); bsr=np.array([c.score for c in rf.face_blendshapes[0]])
        of=fl(img,face_rect_from_pose(lm2,W,H))
        if of:
            of=fl(img,of[2]) or of
            s=np.ptp(reff[:,1])*H
            ef.append(np.mean(np.hypot((of[0][:,0]-reff[:,0])*W,(of[0][:,1]-reff[:,1])*H))/s); ebs.append(np.abs(of[1]-bsr).mean()); nf[0]+=1
print('pose n',len(ep),'det miss',nd,'err1 %.4f err2 %.4f'%(np.median(ep),np.median(ep2)), 'p90 %.4f'%np.percentile(ep2,90))
print('hand matched',nh,'err med %.4f'%(np.median(eh) if eh else -1))
print('face',nf,'err med %.4f bs %.4f'%(np.median(ef) if ef else -1, np.median(ebs) if ebs else -1))
