import sys,io,random,json,hashlib
import numpy as np
from pyboy import PyBoy
W=sys.argv[1]; UP=sys.argv[2]; N=int(sys.argv[3])
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
def run(pb,n=N):
    A=np.zeros((n,8319),dtype=np.uint8)
    for t in range(n): step(pb,SEQ[t]); A[t]=snap(pb)
    return A
def cmp(A,B):
    per=(A!=B).sum(1); idx=np.nonzero(per)[0]
    return 'IDENTICAL' if len(idx)==0 else f'first step {idx[0]+1}, bytes {per[idx[0]]}, final bytes {per[-1]}'
# v30 byte-offset table for the fixed part
F=[('cpu_regs',6,18),('IME',18,19),('ie_delay',19,20),('halted',20,21),('stopped',21,22),('IE',22,23),('int_queued',23,24),('IF',24,25),('cpu_cycles',25,33),('ssh',33,41),('oam_dma',41,53),('VRAM',53,53+0x2000),('OAM',53+0x2000,8405),('LCDC',8405,8406),('palettes',8406,8409),('STAT',8409,8410),('LY',8410,8411),('LYC',8411,8412),('scroll',8412,8416),('objprio',8416,8417),('scanline',8417,9137),('lcd_cgb',9137,9138),('downgraded',9138,9139),('speed_shift',9139,9140),('frame_done',9140,9141),('first_frame',9141,9142),('lcd_reset',9142,9143),('lcd_last_cycles',9143,9151),('lcd_clock',9151,9159),('lcd_clock_target',9159,9167),('next_stat_mode',9167,9168),('stat_transition',9168,9169)]
def locate(raw,wram):
    i=raw.find(wram[:4096],9169); assert i>0
    t=[('sound',9169,i-115201),('screenbuffer',i-115201,i-1),('wy_activated',i-1,i),('WRAM',i,i+8192),('NONIO',i+8192,i+8192+0x60),('IOPORTS',i+8192+0x60,i+8192+0x60+0x4C),('HRAM',i+8192+0xAC,i+8192+0xAC+0x7F)]
    o=i+8491; t+= [('wram_select',o,o+1)]; o+=1
    t+=[('DIV',o,o+1),('TIMA',o+1,o+2),('DIV_counter',o+2,o+4),('TIMA_counter',o+4,o+6),('TMA_TAC',o+6,o+8),('tima_reload',o+8,o+9),('timer_last_cycles',o+9,o+17),('timer_legacy',o+17,o+25)]; o+=25
    t+=[('cart',o,o+4+32768)]; o+=4+32768
    t+=[('joypad',o,o+2)]; o+=2
    t+=[('serial',o,o+36)]; o+=36
    t+=[('sgb',o,len(raw))]
    return F+t
def fdiff(a,b,tbl):
    out={}
    for n,s,e in tbl:
        if a[s:e]!=b[s:e]:
            ds=sum(1 for x,y in zip(a[s:e],b[s:e]) if x!=y)
            val=(int.from_bytes(a[s:e],'little'),int.from_bytes(b[s:e],'little')) if e-s<=8 else f'{ds} bytes differ'
            out[n]=val
    return out
O=open(f'{UP}/has_pokedex.state','rb').read()
OI=open(f'{UP}/init.state','rb').read()
pb=new(); load(pb,O); wram=bytes(pb.memory[0xC000:0xE000]); C=save(pb); FT=run(pb); pb.stop(save=False)
tbl=locate(C,wram)
print('--- A. old-format state loaded after different emulator histories (re-saved fields vs fresh, then trajectory vs fresh)')
hist=[('double load, no ticks',0,False),('1 tick',1,False),('2 ticks',2,False),('7 ticks',7,False),('24 ticks',24,False),('1000 ticks',1000,False),('after 300 steps of init.state',-1,False)]
reused_resaves={}
for name,t,_ in hist:
    pb=new()
    if t==-1:
        load(pb,OI)
        for k in range(300): step(pb,SEQ[k])
    else:
        load(pb,O)
        if t: pb.tick(t,False,False)
    load(pb,O); D=save(pb); reused_resaves[name]=D
    print(name,': resave vs fresh-conv fields',fdiff(C,D,tbl))
    print('    trajectory vs fresh_old:',cmp(FT,run(pb)))
    pb.stop(save=False)
print('--- B. reloading into an emulator with boot-time state (no load before), after boot ticks')
pb=new(); pb.tick(120,False,False); load(pb,O); D=save(pb); print('fresh, 120 boot ticks, then old load: fields',fdiff(C,D,tbl)); print('    trajectory:',cmp(FT,run(pb))); pb.stop(save=False)
print('--- C. attribution: patch fields of converted state C from reused resave D (after 300 steps), load in fresh emulator')
D=reused_resaves['1000 ticks']
pb=new(); load(pb,D); TD=run(pb,60); pb.stop(save=False)
ds=fdiff(C,D,tbl); print('D fields:',ds)
pb=new(); load(pb,C); TC=run(pb,60); pb.stop(save=False)
print('D-in-fresh vs C-in-fresh:',cmp(TC,TD))
# does D loaded fresh reproduce the reused run?
pb=new(); load(pb,O); pb.tick(1000,False,False); load(pb,O); TR=run(pb,60); pb.stop(save=False)
print('D-in-fresh vs real reused run:',cmp(TR,TD))
names=[n for n in ds if n not in('sound','screenbuffer','sgb')]
idx={n:(s,e) for n,s,e in tbl}
for n in names:
    s,e=idx[n]; P=bytearray(C); P[s:e]=D[s:e]
    pb=new(); load(pb,bytes(P)); T=run(pb,60); pb.stop(save=False)
    print(f'  C + only {n} from D: vs C:',cmp(TC,T),'| vs D:',cmp(TD,T))
for n in names:
    s,e=idx[n]; P=bytearray(D); P[s:e]=C[s:e]
    pb=new(); load(pb,bytes(P)); T=run(pb,60); pb.stop(save=False)
    print(f'  D with only {n} from C: vs C:',cmp(TC,T),'| vs D:',cmp(TD,T))
print('--- D. first frames: hRandomAdd/Sub, DIV, LY frame by frame fresh vs reused')
for lab,prep in (('fresh',0),('reused-1000-ticks',1000)):
    pb=new()
    if prep: load(pb,O); pb.tick(prep,False,False)
    load(pb,O); print(lab,'after load LY',pb.memory[0xFF44],'STAT',hex(pb.memory[0xFF41]),'DIV',pb.memory[0xFF04],'R',pb.memory[0xFFD3],pb.memory[0xFFD4],'hFrameCounter',pb.memory[0xFFD5])
    for f in range(4):
        pb.tick(1,False,False); print('  f',f+1,'LY',pb.memory[0xFF44],'STAT',hex(pb.memory[0xFF41]),'DIV',pb.memory[0xFF04],'R',pb.memory[0xFFD3],pb.memory[0xFFD4])
    pb.stop(save=False)
