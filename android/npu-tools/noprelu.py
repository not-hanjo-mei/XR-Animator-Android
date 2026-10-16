import sys, flatbuffers, numpy as np
from ai_edge_litert import schema_py_generated as S
src,dst=sys.argv[1],sys.argv[2]
m=S.ModelT.InitFromObj(S.Model.GetRootAsModel(open(src,'rb').read(),0))
def opcode(code):
    for i,oc in enumerate(m.operatorCodes):
        if max(oc.builtinCode,oc.deprecatedBuiltinCode)==code: return i
    oc=S.OperatorCodeT(); oc.builtinCode=code; oc.deprecatedBuiltinCode=min(code,127); oc.version=1
    m.operatorCodes.append(oc); return len(m.operatorCodes)-1
P=opcode(S.BuiltinOperator.PRELU); MUL=opcode(S.BuiltinOperator.MUL); RELU=opcode(S.BuiltinOperator.RELU); ADD=opcode(S.BuiltinOperator.ADD)
sg=m.subgraphs[0]; ops=[]; amin,amax=9,-9; n=0
for op in sg.operators:
    if op.opcodeIndex!=P: ops.append(op); continue
    x,a=op.inputs; y=op.outputs[0]
    al=np.frombuffer(bytes(bytearray(m.buffers[sg.tensors[a].buffer].data)),np.float32)
    amin=min(amin,al.min()); amax=max(amax,al.max())
    def nt(nm, data=None, shape=None):
        t=S.TensorT(); t.shape=list(shape if shape is not None else sg.tensors[y].shape); t.type=S.TensorType.FLOAT32; t.name=nm.encode()
        b=S.BufferT()
        if data is not None: b.data=np.frombuffer(data.astype(np.float32).tobytes(),np.uint8)
        m.buffers.append(b); t.buffer=len(m.buffers)-1; sg.tensors.append(t); return len(sg.tensors)-1
    ax=nt('prelu_ax_%d'%n); r=nt('prelu_r_%d'%n); rb=nt('prelu_rb_%d'%n)
    om=nt('prelu_1ma_%d'%n, 1-al, sg.tensors[a].shape)
    def mk(code, ins, outs, ot=None, oo=None):
        o=S.OperatorT(); o.opcodeIndex=code; o.inputs=ins; o.outputs=outs
        if ot: o.builtinOptionsType=ot; o.builtinOptions=oo
        return o
    ops+=[mk(MUL,[x,a],[ax],S.BuiltinOptions.MulOptions,S.MulOptionsT()), mk(RELU,[x],[r]),
          mk(MUL,[r,om],[rb],S.BuiltinOptions.MulOptions,S.MulOptionsT()), mk(ADD,[ax,rb],[y],S.BuiltinOptions.AddOptions,S.AddOptionsT())]; n+=1
sg.operators=ops
print('replaced',n,'alpha range',amin,amax)
bld=flatbuffers.Builder(1<<20); bld.Finish(m.Pack(bld), file_identifier=b'TFL3'); open(dst,'wb').write(bld.Output())
