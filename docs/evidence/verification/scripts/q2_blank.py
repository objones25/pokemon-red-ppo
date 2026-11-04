import sys,io
import numpy as np
from pyboy import PyBoy
from PIL import Image
W=sys.argv[1]; UP=sys.argv[2]
ROM=W+'/Pokemon Red.gb'
def new(): return PyBoy(ROM,window='null',sound_emulated=False)
def load(pb,raw): pb.load_state(io.BytesIO(raw))
def save(pb):
    b=io.BytesIO(); pb.save_state(b); return b.getvalue()
def uq(pb):
    a=pb.screen.ndarray[:,:,:3]; return len(np.unique(a.reshape(-1,3),axis=0)), a.copy()
for sn in ['has_pokedex','init','has_pokedex_nballs','fast_text_start']:
    O=open(f'{UP}/{sn}.state','rb').read()
    pb=new(); load(pb,O); C=save(pb); pb.stop(save=False)
    for lab,raw,prep in (('old/fresh',O,None),('conv/fresh',C,None),('old/reused',O,True),('conv/reused',C,True)):
        pb=new()
        if prep: load(pb,raw); pb.tick(500,False,False)
        load(pb,raw); n0,_=uq(pb)
        seq=[]
        for f in range(1,16):
            pb.tick(1,True,False); n,a=uq(pb); seq.append(n)
            if sn=='has_pokedex' and f<=6: Image.fromarray(a.astype('uint8')).resize((320,288),Image.NEAREST).save(f'{W}/img/q2_blank_{lab.replace("/","_")}_f{f}.png')
        # render only frame k
        rk=[]
        for k in (1,2,3,4,5,6,8,24):
            pb2=new()
            if prep: load(pb2,raw); pb2.tick(500,False,False)
            load(pb2,raw)
            if k>1: pb2.tick(k-1,False,False)
            pb2.tick(1,True,False); rk.append((k,uq(pb2)[0])); pb2.stop(save=False)
        print(sn,lab,'colors at load(no tick)=',n0,'each-frame rendered f1..f15=',seq,'render-only-frame-k (k,colors)=',rk,flush=True)
        pb.stop(save=False)
