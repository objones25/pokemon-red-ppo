import sys,io,glob,os,logging
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
from pyboy import PyBoy
W=sys.argv[1]; UP=sys.argv[2]
states=sorted(glob.glob(UP+'/*.state'))+[W+'/project_converted.state']
NAMES={0xB1:'SQUIRTLE',0xB0:'CHARMANDER',0x99:'BULBASAUR',0xFF:'END'}
pb=PyBoy(W+'/Pokemon Red.gb',window='null',sound_emulated=False)
for s in states:
    raw=open(s,'rb').read()
    pb.load_state(io.BytesIO(raw))
    m=pb.memory
    pc=m[0xD163]; sp=[m[0xD164+i] for i in range(7)]
    mon1=[m[0xD16B+i] for i in range(0x2c)]
    bag_n=m[0xD31D]; bag=[(hex(m[0xD31E+2*i]),m[0xD31F+2*i]) for i in range(min(bag_n,20))]
    ev=m[0xD74B]; 
    owned=sum(bin(m[0xD2F7+i]).count('1') for i in range(19)); seen=sum(bin(m[0xD30A+i]).count('1') for i in range(19))
    ownedbits=[i*8+b+1 for i in range(19) for b in range(8) if m[0xD2F7+i]>>b&1]
    seenbits=[i*8+b+1 for i in range(19) for b in range(8) if m[0xD30A+i]>>b&1]
    evcount=sum(bin(m[0xD747+i]).count('1') for i in range(320))
    print(os.path.basename(s),'filebyte0(ver)=',raw[0],'size',len(raw))
    print('  party',pc,[NAMES.get(x,hex(x)) for x in sp],'mon1 species',hex(mon1[0]),'hp',mon1[1]<<8|mon1[2],'lvl',m[0xD18C],'moves',[hex(x) for x in mon1[8:12]],'otid',hex(mon1[12]<<8|mon1[13]))
    print('  map',hex(m[0xD35E]),'xy',m[0xD362],m[0xD361],'bag',bag_n,bag)
    print('  d74b=',format(ev,'08b'),'GOT_POKEDEX bit5',ev>>5&1,'set event bits total',evcount,'owned',owned,ownedbits,'seen',seen,seenbits)
    print('  names',[m[0xD158+i] for i in range(11)],'ball count check (POKE_BALL 0x04):',[q for (i,q) in [(m[0xD31E+2*k],m[0xD31F+2*k]) for k in range(min(bag_n,20))] if i==4])
pb.stop(save=False)
