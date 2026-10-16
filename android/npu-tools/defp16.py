import sys, flatbuffers, numpy as np
from ai_edge_litert import schema_py_generated as S
def run(src,dst):
    buf=open(src,'rb').read()
    m=S.ModelT.InitFromObj(S.Model.GetRootAsModel(buf,0))
    deq=[i for i,oc in enumerate(m.operatorCodes) if max(oc.builtinCode,oc.deprecatedBuiltinCode)==S.BuiltinOperator.DEQUANTIZE]
    sg=m.subgraphs[0]; keep=[]; n=0
    for op in sg.operators:
        if op.opcodeIndex in deq:
            ti,to=op.inputs[0],op.outputs[0]; t=sg.tensors[ti]
            b=m.buffers[t.buffer]
            if t.type==S.TensorType.FLOAT16 and b.data is not None and len(b.data)>0:
                arr=np.frombuffer(bytes(bytearray(b.data)),np.float16).astype(np.float32)
                nb=S.BufferT(); nb.data=np.frombuffer(arr.tobytes(),np.uint8)
                m.buffers.append(nb); sg.tensors[to].buffer=len(m.buffers)-1; n+=1
                continue
        keep.append(op)
    sg.operators=keep
    sd=S.SignatureDefT(); sd.signatureKey='serving_default'; sd.subgraphIndex=0
    def tm(i):
        nm=sg.tensors[i].name; nm=nm.decode() if isinstance(nm,bytes) else nm
        t=S.TensorMapT(); t.name=nm; t.tensorIndex=i; return t
    sd.inputs=[tm(i) for i in sg.inputs]; sd.outputs=[tm(i) for i in sg.outputs]
    m.signatureDefs=[sd]
    bld=flatbuffers.Builder(1<<20); bld.Finish(m.Pack(bld), file_identifier=b'TFL3')
    open(dst,'wb').write(bld.Output()); print(src,'dequant removed',n)
run(sys.argv[1],sys.argv[2])
