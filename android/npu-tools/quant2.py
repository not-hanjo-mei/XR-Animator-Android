# int8 re-quantization with background (negative) crops in calibration so the presence heads keep their low range
import numpy as np, sys, glob
from PIL import Image
from ai_edge_quantizer import quantizer, recipe
from ai_edge_litert.interpreter import Interpreter
M={'pose':('sig/pose_landmarks_detector.tflite',0.5,256),'hand':('sig/hand_landmarks_detector.tflite',0.5,224),'face':('sig/face_landmarks_detector.tflite',0.0,256)}
which=sys.argv[1]; path,thr,S=M[which]
X=np.load(f'calib/{which}.npy'); sc=np.load(f'calib/{which}_score.npy'); X=X[sc>thr]
pos=X[:int(len(X)*0.8)][:120]
rng=np.random.default_rng(1); neg=[]
for f in sorted(glob.glob('calib/img/*.jpg'))[300:340]:
    im=Image.open(f).convert('RGB'); w,h=im.size; s=int(min(w,h)*rng.uniform(0.15,0.6)); x=rng.integers(0,w-s); y=rng.integers(0,h-s)
    neg.append(np.asarray(im.crop((x,y,x+s,y+s)).resize((S,S)),np.float32)/255)
cal=list(pos)+neg; rng.shuffle(cal)
inp=Interpreter(model_path=path).get_input_details()[0]['name']
qt=quantizer.Quantizer(path); qt.load_quantization_recipe(recipe.static_wi8_ai8())
cr=qt.calibrate({'serving_default':[{inp:a[None]} for a in cal]})
qt.quantize(cr).export_model(f'q/{which}_i8n.tflite', overwrite=True); print('ok',which)
