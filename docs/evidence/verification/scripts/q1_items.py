import sys,io
from pyboy import PyBoy
from PIL import Image
W=sys.argv[1]; UP=sys.argv[2]
pb=PyBoy(W+'/Pokemon Red.gb',window='null',sound_emulated=False)
ims=[]
for n,p in (('has_pokedex',UP+'/has_pokedex.state'),('has_pokedex_nballs',UP+'/has_pokedex_nballs.state'),('project_converted',W+'/project_converted.state')):
    pb.load_state(io.BytesIO(open(p,'rb').read())); pb.tick(120,True,False)
    for b,f in (('start',60),('down',30),('down',30),('a',90)):
        pb.button(b,8); pb.tick(f,True,False)
    im=pb.screen.image.convert('RGB'); im.save(f'{W}/img/q1_{n}_item_screen.png'); ims.append(im)
m=Image.new('RGB',(3*324,288),'white')
for i,im in enumerate(ims): m.paste(im.resize((320,288),Image.NEAREST),(i*324,0))
m.save(f'{W}/img/q1_item_screens_montage.png'); pb.stop(save=False)
