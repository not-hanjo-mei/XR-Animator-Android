import numpy as np, sys, traceback
from ai_edge_quantizer import quantizer, recipe
from ai_edge_litert.interpreter import Interpreter
M={'pose':('sig/pose_landmarks_detector.tflite',0.5),
   'hand':('sig/hand_landmarks_detector.tflite',0.5),
   'face':('sig/face_landmarks_detector.tflite',0.0)}
which=sys.argv[1]; rname=sys.argv[2]
path,thr=M[which]
X=np.load(f'calib/{which}.npy'); S=np.load(f'calib/{which}_score.npy'); X=X[S>thr]
n=len(X); cal=X[:int(n*0.8)]
inp=Interpreter(model_path=path).get_input_details()[0]['name']
qt=quantizer.Quantizer(path)
qt.load_quantization_recipe(getattr(recipe,rname)())
sig=None
try:
    from ai_edge_quantizer.utils import tfl_interpreter_utils as u
except Exception: pass
data={'serving_default':[{inp:a[None]} for a in cal[:120]]}
try:
    cr=qt.calibrate(data)
except Exception as e:
    traceback.print_exc(limit=3); sys.exit(1)
qt.quantize(cr).export_model(f'q/{which}_{rname}.tflite', overwrite=True)
print('ok', which, rname, n)
