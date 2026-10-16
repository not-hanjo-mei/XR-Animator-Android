import os
import json, numpy as np, pipe
from PIL import Image
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from pipe import *
A=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../') + '/app/src/main/assets/models/'
P=json.load(open('../calib/persons.json'))
files=list(dict.fromkeys(p['file'] for p in P if p['bbox'][3]>250))[:60]
mpH=vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=A+'hand_landmarker.task'),num_hands=2))
hl=HandLmk(); data=[]
for f in files:
    im=np.asarray(Image.open('../calib/img/'+f).convert('RGB')); H,W=im.shape[:2]
    rh=mpH.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=im))
    for hh in rh.hand_landmarks: data.append((im,np.array([[l.x*W,l.y*H] for l in hh])))
def run(tag, rotsign=1, rng=(0,1), scale=2.0):
    e=[]
    for im,rf in data:
        H,W=im.shape[:2]; img=im.astype(np.float32)/255; s=np.ptp(rf[:,1])+1
        cx,cy,sz,r=hand_rect_from_lm(rf); sz=sz/2.0*scale; r*=rotsign; rect=(cx,cy,sz,r)
        x=warp(img,rect,224)*(rng[1]-rng[0])+rng[0]
        lm,pres,hand,world=hl.net(x)
        lm=lm.reshape(21,3); p=np.array([unproject(a/224,b/224,rect) for a,b,_ in lm])
        e.append(np.mean(np.hypot(*(p-rf).T))/s)
    print(tag, np.median(e))
run('base'); run('rot-',-1); run('rng-11',1,(-1,1)); run('scale2.4',scale=2.4); run('scale1.7',scale=1.7)
