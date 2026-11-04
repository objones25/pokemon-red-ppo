import sys,io,os,hashlib
sys.path.insert(0,sys.argv[1]+'/scripts')
from stateparse import parse
from pyboy import PyBoy
W=sys.argv[1]; UP=sys.argv[2]
sym={}
for l in open(W+'/src/sym_dl/pokered_symbols_branch.sym'):
    if l[:3]=='00:' :
        a,n=l.split()[:2]; a=int(a[3:],16)
        if a>=0xC000: sym.setdefault(a,n)
keys=sorted(sym)
import bisect
def name(a):
    i=bisect.bisect_right(keys,a)-1
    return f'{sym[keys[i]]}+{a-keys[i]}' if i>=0 else '?'
def mem(pb,a,b): return bytes(pb.memory[a:b])
def snap(pb):
    return {'WRAM':mem(pb,0xC000,0xE000),'HRAM':mem(pb,0xFF80,0xFFFF),'VRAM':mem(pb,0x8000,0xA000),'OAM':mem(pb,0xFE00,0xFEA0),'IO':mem(pb,0xFF00,0xFF80),'IE':mem(pb,0xFFFF,0x10000)}
old=open(UP+'/has_pokedex.state','rb').read(); new=open(W+'/project_converted.state','rb').read(); nb=open(UP+'/has_pokedex_nballs.state','rb').read()
def fresh(raw):
    pb=PyBoy(W+'/Pokemon Red.gb',window='null',sound_emulated=False); pb.load_state(io.BytesIO(raw)); return pb
A=fresh(old);sa=snap(A);B=fresh(new);sb=snap(B);C=fresh(nb);sc=snap(C)
print('--- has_pokedex(v7) vs project converted(v30), memory right after load_state, fresh emulators')
for k in sa:
    dif=[i for i,(x,y) in enumerate(zip(sa[k],sb[k])) if x!=y]
    print(k,len(sa[k]),'bytes; differ',len(dif),'sha',hashlib.sha1(sa[k]).hexdigest()[:10],hashlib.sha1(sb[k]).hexdigest()[:10],[hex(i) for i in dif][:12])
print('--- has_pokedex vs has_pokedex_nballs')
for k,base in (('WRAM',0xC000),('HRAM',0xFF80),('VRAM',0x8000),('OAM',0xFE00),('IO',0xFF00)):
    dif=[i for i,(x,y) in enumerate(zip(sa[k],sc[k])) if x!=y]
    print(k,'differ',len(dif))
    if k in('WRAM','HRAM'):
        for i in dif[:60]: print('   ',hex(base+i),name(base+i),hex(sa[k][i]),'->',hex(sc[k][i]))
# my conversion
buf=io.BytesIO();A.save_state(buf);mine=buf.getvalue()
print('my conversion(fresh load v7 then save) == project state:',mine==new,len(mine),len(new))
if mine!=new:
    dif=[i for i,(x,y) in enumerate(zip(mine,new)) if x!=y];print(' differing bytes',len(dif),dif[:40])
buf=io.BytesIO();B.save_state(buf);print('project state load+resave == file:',buf.getvalue()==new)
po=parse(old);pn=parse(new,sa['WRAM']);pm=parse(mine,sa['WRAM'])
print('--- field compare v7 file vs v30 file (common fields); fields only in v30 printed with value')
for k in pn:
    if k in po:
        if po[k]!=pn[k]:
            v=lambda x:(x if not isinstance(x,bytes) else f'{len(x)}B sha {hashlib.sha1(x).hexdigest()[:8]}')
            extra=''
            if isinstance(po[k],bytes) and len(po[k])==len(pn[k]):
                ds=[i for i,(x,y) in enumerate(zip(po[k],pn[k])) if x!=y]; extra=f' {len(ds)} bytes differ at {ds[:12]}'
            print(' DIFF',k,v(po[k]),v(pn[k]),extra)
    else:
        v=pn[k]; print(' v30only',k,v if not isinstance(v,bytes) else (f'{len(v)}B nonzero={sum(1 for x in v if x)} sha {hashlib.sha1(v).hexdigest()[:8]}'))
print('common equal:',[k for k in pn if k in po and po[k]==pn[k]])
print('registers v7',{k:(hex(po[k])) for k in 'A F B C D E HL SP PC IME halted IE LCDC STAT LY LYC DIV TIMA DIV_counter TIMA_counter TMA TAC'.split()})
for p in (A,B,C): p.stop(save=False)
