import sys,io,re,random
import numpy as np
from pyboy import PyBoy
W,UP=sys.argv[1:3]
names={}
v=0
for l in open(W+'/src/pokered_dl/pokered/constants/event_constants.asm'):
    s=l.split(';')[0].strip()
    if s=='const_def': v=0
    elif (m:=re.match(r'const_next\s+(.+)',s)): v=eval(m.group(1).replace('$','0x'))
    elif (m:=re.match(r'const_skip(?:\s+(.+))?$',s)): v+=int(m.group(1) or 1)
    elif (m:=re.match(r'const\s+(\w+)',s)): names[v]=m.group(1); v+=1
pb=PyBoy(W+'/Pokemon Red.gb',window='null',sound_emulated=False)
for sn in ('has_pokedex','has_pokedex_nballs'):
    pb.load_state(io.BytesIO(open(f'{UP}/{sn}.state','rb').read())); m=pb.memory
    lsb=[(i*8+b) for i in range(320) for b in range(8) if m[0xD747+i]>>b&1]
    msb=[(i*8+7-b) for i in range(320) for b in range(8) if m[0xD747+i]>>b&1]
    print(sn,'LSB-first:',[(hex(n),names.get(n,'UNDEFINED')) for n in lsb]); print('  MSB-first:',[(hex(n),names.get(n,'UNDEFINED')) for n in msb])
    print('  wOptions D355',format(m[0xD355],'08b'),'textspeed nibble',m[0xD355]&15,'money',[hex(m[0xD347+i]) for i in range(3)],'bytes>=311 nonzero',[hex(m[0xD747+i]) for i in range(311,320)])
pb.stop(save=False)
# WRAM-only fresh vs reused on init (1500 steps)
BUTTONS=['a','b','start','up','down','left','right']; rng=random.Random(20261103); SEQ=[rng.choice(BUTTONS) for _ in range(3000)]
def run(pb):
    A=np.zeros((3000,8319),np.uint8)
    for t,b in enumerate(SEQ):
        pb.button_press(b); pb.tick(8,False,False); pb.button_release(b); pb.tick(15,False,False); pb.tick(1,True,False)
        A[t]=np.array(pb.memory[0xC000:0xE000]+pb.memory[0xFF80:0xFFFF],np.uint8)
    return A
O=open(UP+'/init.state','rb').read(); pb=PyBoy(W+'/Pokemon Red.gb',window='null',sound_emulated=False); pb.load_state(io.BytesIO(O)); A=run(pb); pb.load_state(io.BytesIO(O)); B=run(pb); pb.stop(save=False)
ne=A!=B; w=ne[:,:8192].any(1); h=ne[:,8192:].any(1)
print('init fresh vs reused WRAM-only: first',int(np.argmax(w))+1 if w.any() else None,'steps differing in first 1500:',int(w[:1500].sum()),'in 3000:',int(w.sum()),' HRAM steps',int(h.sum()))
hx=np.nonzero(ne[:,8192:].any(0))[0]; print(' HRAM addrs differing ever',[hex(0xFF80+int(i)) for i in hx])
wx=np.nonzero(ne[:,:8192].any(0))[0]; print(' WRAM addrs differing ever',[hex(0xC000+int(i)) for i in wx][:30])
