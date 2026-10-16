import sys, flatbuffers
from ai_edge_litert import schema_py_generated as S
def addsig(src,dst):
    buf=open(src,'rb').read()
    m=S.ModelT.InitFromObj(S.Model.GetRootAsModel(buf,0))
    sg=m.subgraphs[0]
    sd=S.SignatureDefT(); sd.signatureKey='serving_default'; sd.subgraphIndex=0
    def tm(i):
        t=S.TensorMapT(); t.name=sg.tensors[i].name.decode() if isinstance(sg.tensors[i].name,bytes) else sg.tensors[i].name; t.tensorIndex=i; return t
    sd.inputs=[tm(i) for i in sg.inputs]; sd.outputs=[tm(i) for i in sg.outputs]
    m.signatureDefs=[sd]
    b=flatbuffers.Builder(1024); b.Finish(m.Pack(b), file_identifier=b'TFL3')
    open(dst,'wb').write(b.Output())
if __name__=='__main__': addsig(sys.argv[1],sys.argv[2])
