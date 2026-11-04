import sys,io,os
from pyboy import PyBoy
from PIL import Image
W=sys.argv[1]; UP=sys.argv[2]
states={'has_pokedex':UP+'/has_pokedex.state','has_pokedex_nballs':UP+'/has_pokedex_nballs.state','project_converted':W+'/project_converted.state','init':UP+'/init.state'}
SEQ=[('settled',None,120),('start_menu','start',60),('pokedex_list','a',90),('back1','b',60),('down_on_pokemon','down',30),('party_screen','a',90),('mon_submenu','a',60),('mon_stats','a',90)]
SEQ_INIT=[('settled',None,240),('start_menu','start',60),('a_top_item','a',90)]
pb=PyBoy(W+'/Pokemon Red.gb',window='null',sound_emulated=False)
for name,path in states.items():
    pb.load_state(io.BytesIO(open(path,'rb').read()))
    imgs=[]
    for lab,btn,fr in (SEQ_INIT if name=='init' else SEQ):
        if btn: pb.button(btn,8)
        pb.tick(fr,True,False)
        im=pb.screen.image.convert('RGB')
        p=f'{W}/img/q1_{name}_{lab}.png'; im.save(p); imgs.append(im)
    m=Image.new('RGB',(4*162*2,((len(imgs)+3)//4)*146*2),'white')
    for i,im in enumerate(imgs):
        m.paste(im.resize((320,288),Image.NEAREST),((i%4)*324,(i//4)*292))
    m.save(f'{W}/img/q1_{name}_montage.png'); print(name,'wrote',len(imgs))
pb.stop(save=False)
