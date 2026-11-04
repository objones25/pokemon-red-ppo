import sys,io
from pyboy import PyBoy
W,UP=sys.argv[1:3]
pb=PyBoy(W+'/Pokemon Red.gb',window='null',sound_emulated=False)
pb.load_state(io.BytesIO(open(UP+'/has_pokedex_nballs.state','rb').read())); pb.tick(120,True,False)
pb.button('start',8); pb.tick(60,True,False); pb.screen.image.convert('RGB').save(W+'/img/q1_nballs_start.png')
pb.button('a',8); pb.tick(90,True,False); pb.screen.image.convert('RGB').resize((320,288),0).save(W+'/img/q1_nballs_item_screen.png')
pb.stop(save=False)
