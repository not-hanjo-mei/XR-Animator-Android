import json, numpy as np, math
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
exec(open('make_crops.py').read().split('pose,hand,face')[0])
it=Interpreter(model_path='models/hand_landmarker/hand_landmarks_detector.tflite'); it.allocate_tensors()
inp=it.get_input_details()[0]['index']; o=it.get_output_details()[1]['index']
def sc(a): it.set_tensor(inp,a[None]); it.invoke(); return float(it.get_tensor(o).ravel()[0])
for off,ext,sz in ((-math.pi/2,0.35,1.1),(math.pi/2,0.35,1.1),(-math.pi/2,0.5,1.3),(math.pi/2,0.5,1.3),(-math.pi/2,0.25,0.9),(math.pi/2,0.6,1.6)):
  r=[]
  for p in P[:120]:
    im=Image.open('calib/img/'+p['file']).convert('RGB'); kp=np.array(p['kp'],np.float32).reshape(17,3)
    for e,wi in ((7,9),(8,10)):
        if kp[e,2]>0 and kp[wi,2]>0:
            fa=kp[wi,:2]-kp[e,:2]; L=np.linalg.norm(fa)
            if L<30: continue
            c=kp[wi,:2]+fa*ext; ang=math.atan2(fa[1],fa[0])+off
            r.append(sc(crop(im,c[0],c[1],L*sz,ang,224)))
  r=np.array(r); print(off,ext,sz,len(r),np.median(r),(r>0.5).mean())
