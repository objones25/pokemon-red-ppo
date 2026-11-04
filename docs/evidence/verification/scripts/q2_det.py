import sys,io,os,json,random,time,bisect,hashlib
import numpy as np
from pyboy import PyBoy
W=sys.argv[1]; UP=sys.argv[2]; STATES=sys.argv[3].split(','); N=int(sys.argv[4]); OUT=sys.argv[5]
ROM=W+'/Pokemon Red.gb'
BUTTONS=['a','b','start','up','down','left','right']
rng=random.Random(20261103); SEQ=[rng.choice(BUTTONS) for _ in range(N)]
print('seq sha',hashlib.sha1(','.join(SEQ).encode()).hexdigest()[:10],SEQ[:12],flush=True)
sym={}
for l in open(W+'/src/sym_dl/pokered_symbols_branch.sym'):
    if l[:3]=='00:':
        a,n=l.split()[:2]; a=int(a[3:],16)
        if a>=0xC000: sym.setdefault(a,n)
keys=sorted(sym)
def name(a):
    i=bisect.bisect_right(keys,a)-1
    return f'{sym[keys[i]]}+{a-keys[i]}'
def addr(i): return 0xC000+i if i<8192 else 0xFF80+i-8192
def new(): return PyBoy(ROM,window='null',sound_emulated=False)
def load(pb,raw): pb.load_state(io.BytesIO(raw))
def snap(pb): return np.array(pb.memory[0xC000:0xE000]+pb.memory[0xFF80:0xFFFF],dtype=np.uint8)
def step(pb,b):
    pb.button_press(b); pb.tick(8,False,False); pb.button_release(b); pb.tick(15,False,False); pb.tick(1,True,False)
def run(pb,a=0,b=N,arr=None,hook=None):
    if arr is None: arr=np.zeros((N,8319),dtype=np.uint8)
    for t in range(a,b):
        step(pb,SEQ[t]); arr[t]=snap(pb)
        if hook: hook(t)
    return arr
def cmp(A,B,a=0):
    ne=(A[a:]!=B[a:])
    per=ne.sum(1)
    idx=np.nonzero(per)[0]
    if len(idx)==0: return {'identical':True,'from_step':a}
    f=int(idx[0])+a; ad=np.nonzero(A[f]!=B[f])[0]
    fi=A[-1]!=B[-1]; fa=np.nonzero(fi)[0]
    r={'identical':False,'first_step':f+1,'n_bytes_first':len(ad),'first_addrs':[(hex(addr(int(i))),name(addr(int(i))),int(A[f][i]),int(B[f][i])) for i in ad[:25]],
       'steps_differ':len(idx),'of':N-a,'final_n_bytes':len(fa),'final_addrs_sample':[hex(addr(int(i))) for i in fa[:12]],
       'median_bytes':int(np.median(per[idx])),'game_final':[gi(A[-1]),gi(B[-1])]}
    return r
def gi(s):
    g=lambda a:int(s[a-0xC000])
    return {'map':g(0xD35E),'y':g(0xD361),'x':g(0xD362),'party':g(0xD163),'events':int(sum(bin(int(x)).count('1') for x in s[0xD747-0xC000:0xD887-0xC000])),'money':[g(0xD347),g(0xD348),g(0xD349)]}
res={}
def rec(key,r):
    res[key]=r
    print(key, ('IDENTICAL' if r.get('identical') else f"DIVERGE step {r['first_step']} bytes {r['n_bytes_first']} final {r['final_n_bytes']} {r['first_addrs'][:6]} game {r['game_final']}"),flush=True)
    json.dump(res,open(OUT,'w'),indent=1)
t0=time.time()
for sname in STATES:
    O=open(f'{UP}/{sname}.state','rb').read()
    pb=new(); load(pb,O); b=io.BytesIO(); pb.save_state(b); C=b.getvalue(); pb.stop(save=False)
    if sname=='has_pokedex':
        assert C==open(W+'/project_converted.state','rb').read()
    print('==',sname,'converted len',len(C),time.time()-t0,flush=True)
    # fresh + old
    pb=new(); load(pb,O); F_O=run(pb); print(' t',time.time()-t0,flush=True)
    # same emulator repeated, old
    load(pb,O); R_O1=run(pb); rec(f'{sname}: reused_old#1 vs fresh_old',cmp(F_O,R_O1))
    load(pb,O); R_O2=run(pb); rec(f'{sname}: reused_old#2 vs reused_old#1 (same emu repeated)',cmp(R_O1,R_O2))
    rec(f'{sname}: reused_old#2 vs fresh_old',cmp(F_O,R_O2))
    pb.stop(save=False)
    pb=new(); load(pb,O); F_O2=run(pb); rec(f'{sname}: fresh_old#2 vs fresh_old',cmp(F_O,F_O2)); pb.stop(save=False)
    # converted
    pb=new(); load(pb,C); F_C=run(pb); rec(f'{sname}: fresh_conv vs fresh_old',cmp(F_O,F_C))
    load(pb,C); R_C1=run(pb); rec(f'{sname}: reused_conv#1 vs fresh_conv',cmp(F_C,R_C1))
    load(pb,C); R_C2=run(pb); rec(f'{sname}: reused_conv#2 vs reused_conv#1 (same emu repeated)',cmp(R_C1,R_C2))
    load(pb,O); R_OafterC=run(pb); rec(f'{sname}: old_loaded_in_emu_after_conv_runs vs fresh_old',cmp(F_O,R_OafterC))
    pb.stop(save=False)
    # mid-run
    M=N//2
    pb=new(); load(pb,C); saved={}
    A=np.zeros((N,8319),dtype=np.uint8); run(pb,0,M,A)
    b=io.BytesIO(); pb.save_state(b); S=b.getvalue()
    load(pb,S); B=A.copy(); run(pb,M,N,B); rec(f'{sname}: conv, midrun save+load same emu @ {M} vs uninterrupted',cmp(F_C,B))
    # rewind: emulator has already gone to N; load S again
    load(pb,S); B2=A.copy(); run(pb,M,N,B2); rec(f'{sname}: conv, rewind to {M} after reaching end vs uninterrupted',cmp(F_C,B2))
    pb.stop(save=False)
    pb=new(); load(pb,S); B3=A.copy(); run(pb,M,N,B3); rec(f'{sname}: conv, midrun state loaded in fresh emu @ {M} vs uninterrupted',cmp(F_C,B3)); pb.stop(save=False)
    # old-start mid-run in same emu (save v30 from a run begun at old-format state)
    pb=new(); load(pb,O); A=np.zeros((N,8319),dtype=np.uint8); run(pb,0,M,A)
    b=io.BytesIO(); pb.save_state(b); load(pb,io.BytesIO(b.getvalue()).getvalue()); B=A.copy(); run(pb,M,N,B); rec(f'{sname}: old, midrun save+load same emu @ {M} vs uninterrupted',cmp(F_O,B)); pb.stop(save=False)
    # save+restore every 100 steps
    pb=new(); load(pb,C)
    def hk(t):
        if (t+1)%100==0:
            bb=io.BytesIO(); pb.save_state(bb); load(pb,bb.getvalue())
    B=run(pb,hook=hk); rec(f'{sname}: conv, save+load every 100 steps vs uninterrupted',cmp(F_C,B)); pb.stop(save=False)
    print(' t',time.time()-t0,flush=True)
print('DONE',time.time()-t0)
