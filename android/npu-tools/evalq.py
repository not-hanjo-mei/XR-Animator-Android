import numpy as np
from ai_edge_litert.interpreter import Interpreter
M={'pose':('sig/pose_landmarks_detector.tflite',0.5,5,39,256),'hand':('sig/hand_landmarks_detector.tflite',0.5,3,21,224),'face':('sig/face_landmarks_detector.tflite',0.0,3,478,256)}
def run(f,X):
    it=Interpreter(model_path=f); it.allocate_tensors()
    d=it.get_input_details()[0]; o=it.get_output_details()
    # find the landmark output = first output with size matching
    res=[]
    for a in X:
        x=a[None]
        if d['dtype']!=np.float32:
            s,z=d['quantization']; info=np.iinfo(d['dtype']); x=np.clip(np.round(x/s+z),info.min,info.max).astype(d['dtype'])
        it.set_tensor(d['index'],x); it.invoke()
        outs=[]
        for od in o:
            v=it.get_tensor(od['index']).astype(np.float32)
            if od['dtype']!=np.float32:
                s,z=od['quantization']; v=(v-z)*s
            outs.append(v.ravel())
        res.append(outs)
    return res
for k,(f,thr,stride,npt,sz) in M.items():
    X=np.load(f'calib/{k}.npy'); S=np.load(f'calib/{k}_score.npy'); X=X[S>thr]; X=X[int(len(X)*0.8):]
    ref=run(f,X)
    L=lambda outs: next(v for v in outs if v.size==stride*npt).reshape(npt,stride)[:,:2]
    for r in ('static_wi8_ai8','static_wi8_ai16'):
        q=run(f'q/{k}_{r}.tflite',X)
        e=np.array([np.linalg.norm(L(a)-L(b),axis=1).mean()/sz*100 for a,b in zip(ref,q)])
        print(f'{k:5s} {r:16s} n={len(X)} mean err {e.mean():.2f}% of crop, p90 {np.percentile(e,90):.2f}%')
