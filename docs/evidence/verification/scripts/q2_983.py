import sys,io
import numpy as np
from pyboy import PyBoy
W,UP=sys.argv[1:3]
B=("down","left","right","up","a","b","start")
acts=np.random.default_rng(0).integers(0,7,1500)
def run(pb):
    A=np.zeros((1500,8319),np.uint8)
    for t,a in enumerate(acts):
        b=B[int(a)]; pb.button_press(b); pb.tick(8,False,False); pb.button_release(b); pb.tick(15,False,False); pb.tick(1,True,False)
        A[t]=np.array(pb.memory[0xC000:0xE000]+pb.memory[0xFF80:0xFFFF],np.uint8)
    return A
O=open(UP+'/init.state','rb').read()
pb=PyBoy(W+'/Pokemon Red.gb',window='null',sound_emulated=False)
pb.load_state(io.BytesIO(O)); X=run(pb); pb.load_state(io.BytesIO(O)); Y=run(pb); pb.stop(save=False)
ne=X!=Y; w=ne[:,:8192].any(1); h=ne[:,8192:].any(1)
print('WRAM only: first step',int(np.argmax(w))+1,'steps differing',int(w.sum()),'/1500')
print('WRAM+HRAM: first step',int(np.argmax(w|h))+1,'steps differing',int((w|h).sum()),'/1500; bytes at step1',[hex(0xFF80+int(i)) for i in np.nonzero(ne[0,8192:])[0]])
f=int(np.argmax(w)); print('WRAM addrs at first WRAM diff',[hex(0xC000+int(i)) for i in np.nonzero(ne[f,:8192])[0]])
