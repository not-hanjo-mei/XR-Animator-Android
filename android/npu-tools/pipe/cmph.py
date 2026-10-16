import os
import json, numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from pipe import *
A=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../') + '/app/src/main/assets/models/'
P=json.load(open('../calib/persons.json'))
files=list(dict.fromkeys(p['file'] for p in P if p['bbox'][3]>250))[:60]
mpH=vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=A+'hand_landmarker.task'),num_hands=2))
hl=HandLmk(); e1=[];e2=[];hs=[]
for f in files:
    im=np.asarray(Image.open('../calib/img/'+f).convert('RGB')); H,W=im.shape[:2]; img=im.astype(np.float32)/255
    rh=mpH.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=im))
    for hh,hd in zip(rh.hand_landmarks,rh.handedness):
        rf=np.array([[l.x*W,l.y*H] for l in hh]); s=np.ptp(rf[:,1])+1
        r=hand_rect_from_lm(rf); o=hl(img,r)
        if o is None: continue
        p=o[0][:,:2]*[W,H]; e1.append(np.mean(np.hypot(*(p-rf).T))/s)
        o2=hl(img,o[3]); 
        if o2: p=o2[0][:,:2]*[W,H]; e2.append(np.mean(np.hypot(*(p-rf).T))/s)
        hs.append((hd[0].category_name, round(hd[0].score,2), round(o[2],2)))
print(len(e1),np.median(e1),np.median(e2)); print(hs[:12])
