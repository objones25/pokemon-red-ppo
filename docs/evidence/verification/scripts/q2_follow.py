import sys,io,random,hashlib
import numpy as np
from pyboy import PyBoy
W=sys.argv[1]; UP=sys.argv[2]; mode=sys.argv[3]
ROM=W+'/Pokemon Red.gb'
BUTTONS=['a','b','start','up','down','left','right']
rng=random.Random(20261103); SEQ=[rng.choice(BUTTONS) for _ in range(3000)]
def new(): return PyBoy(ROM,window='null',sound_emulated=False)
def load(pb,raw): pb.load_state(io.BytesIO(raw))
def save(pb):
    b=io.BytesIO(); pb.save_state(b); return b.getvalue()
def snap(pb): return np.array(pb.memory[0xC000:0xE000]+pb.memory[0xFF80:0xFFFF],dtype=np.uint8)
def step(pb,b):
    pb.button_press(b); pb.tick(8,False,False); pb.button_release(b); pb.tick(15,False,False); pb.tick(1,True,False)
def run(pb,n=3000):
    A=np.zeros((n,8319),dtype=np.uint8)
    for t in range(n): step(pb,SEQ[t]); A[t]=snap(pb)
    return A
def cmp(A,B):
    per=(A!=B).sum(1); idx=np.nonzero(per)[0]
    return 'IDENTICAL' if len(idx)==0 else f'first step {idx[0]+1}, bytes {per[idx[0]]}, final bytes {per[-1]}, steps differing {len(idx)}/{len(A)}'
O=open(f'{UP}/has_pokedex.state','rb').read()
if mode=='proc':
    pb=new(); load(pb,O); A=run(pb); pb.stop(save=False); print('process trajectory sha1',hashlib.sha1(A.tobytes()).hexdigest()); sys.exit()
pb=new(); load(pb,O); C=save(pb); FO=run(pb); pb.stop(save=False)
print('--- old-format pairwise by history (same state, same actions)')
T={}
for k in (1,2,7,24,1000):
    pb=new(); load(pb,O); pb.tick(k,False,False); load(pb,O); T[k]=run(pb); pb.stop(save=False)
ks=list(T)
for i,a in enumerate(ks):
    for b in ks[i+1:]: print(f'reused after {a} ticks vs after {b} ticks:',cmp(T[a],T[b]))
print('--- converted loaded after varied histories vs fresh')
for k in (1,2,7,1000):
    pb=new(); load(pb,O); pb.tick(k,False,False); load(pb,C); print(k,'old-then-ticks-then-conv:',cmp(FO,run(pb))); pb.stop(save=False)
print('--- group attribution, 3000 steps. C=converted, D=re-save of old loaded after 1000 ticks')
pb=new(); load(pb,O); pb.tick(1000,False,False); load(pb,O); D=save(pb); TD=run(pb); pb.stop(save=False)
OFF={'cpu_cycles':(25,33),'first_frame':(9141,9142),'lcd_reset':(9142,9143),'lcd_last_cycles':(9143,9151),'lcd_clock':(9151,9159),'lcd_clock_target':(9159,9167),'next_stat_mode':(9167,9168)}
wram=bytes(FO[0][:0]) 
pb=new(); load(pb,O); w=bytes(pb.memory[0xC000:0xE000]); pb.stop(save=False)
i=C.find(w[:4096],9169); assert i>0
OFF['timer_last_cycles']=(i+8491+1+9,i+8491+1+17)
CY=['cpu_cycles','lcd_last_cycles','timer_last_cycles']; LCD=['first_frame','lcd_reset','lcd_clock','lcd_clock_target','next_stat_mode']
def patch(base,src,fs):
    P=bytearray(base)
    for f in fs: s,e=OFF[f]; P[s:e]=src[s:e]
    return bytes(P)
for lab,fs in (('cycle counters only',CY),('LCD phase only',LCD),('cycles+LCD',CY+LCD)):
    pb=new(); load(pb,patch(C,D,fs)); T2=run(pb); pb.stop(save=False)
    print(f'C + {lab} from D: vs C {cmp(FO,T2)} | vs D {cmp(TD,T2)}')
for f in LCD:
    fs=CY+[x for x in LCD if x!=f]
    pb=new(); load(pb,patch(C,D,fs)); T2=run(pb); pb.stop(save=False)
    print(f'C + cycles+LCD except {f}: vs D {cmp(TD,T2)}')
print('--- gameplay divergence, old fresh vs old reused (after a full run)')
for sn in ['has_pokedex','init','has_pokedex_nballs','fast_text_start']:
    O2=open(f'{UP}/{sn}.state','rb').read()
    pb=new(); load(pb,O2); A=run(pb); load(pb,O2); B=run(pb); pb.stop(save=False)
    ne=A!=B; mask=ne.copy(); mask[:,0xFFD3-0xFF80+8192]=False; mask[:,0xFFD4-0xFF80+8192]=False
    nr=np.nonzero(mask.any(1))[0]
    pos=[0xD35E-0xC000,0xD361-0xC000,0xD362-0xC000]
    pd=np.nonzero((A[:,pos]!=B[:,pos]).any(1))[0]
    per=ne.sum(1)
    print(sn,'steps differing',int((per>0).sum()),'median',int(np.median(per)),'max',int(per.max()),'first non-RNG step',int(nr[0])+1 if len(nr) else None,'first map/pos step',int(pd[0])+1 if len(pd) else None,'final',int(per[-1]))
    if len(nr):
        f=nr[0]; print('   first non-RNG addrs',[hex(0xC000+int(i) if i<8192 else 0xFF80+int(i)-8192) for i in np.nonzero(mask[f])[0][:12]])
