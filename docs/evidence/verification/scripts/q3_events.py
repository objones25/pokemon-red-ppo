import re,sys
P=sys.argv[1]
v=0;ev={};order=[]
for ln,l in enumerate(open(P+'/constants/event_constants.asm'),1):
    s=l.split(';')[0].strip()
    if s=='const_def': v=0
    elif (m:=re.match(r'const_next\s+(.+)',s)):
        e=m.group(1).replace('$','0x'); v=eval(e)
    elif (m:=re.match(r'const_skip(?:\s+(.+))?$',s)):
        v+=int(m.group(1) or 1)
    elif (m:=re.match(r'const\s+(\w+)',s)):
        ev[m.group(1)]=(v,ln);order.append(m.group(1));v+=1
    elif 'NUM_EVENTS' in s: print('NUM_EVENTS',hex(v),v,'bits',v/8,'bytes line',ln)
print('defined',len(ev))
top=sorted(ev.items(),key=lambda kv:-kv[1][0])[:6]
for k,(n,ln) in top: print(k,hex(n),n,'byte',n//8,'bit',n%8,'addr',hex(0xd747+n//8),'line',ln)
n,ln=ev['EVENT_GOT_POKEDEX'];print('EVENT_GOT_POKEDEX',hex(n),n,'byte',n//8,'bit',n%8,'addr',hex(0xd747+n//8),'line',ln)
for k in ['EVENT_FOLLOWED_OAK_INTO_LAB','EVENT_OAK_ASKED_TO_CHOOSE_MON','EVENT_GOT_STARTER','EVENT_BATTLED_RIVAL_IN_OAKS_LAB','EVENT_GOT_OAKS_PARCEL','EVENT_OAK_GOT_PARCEL','EVENT_GOT_TOWN_MAP','EVENT_BEAT_ARTICUNO']:
    if k in ev: n,ln=ev[k];print(k,hex(n),n,'byte',n//8,'bit',n%8,'addr',hex(0xd747+n//8),'line',ln)
byte_counts={}
for k,(n,_) in ev.items(): byte_counts[n//8]=byte_counts.get(n//8,0)+1
print('bytes with defined flags:',len(byte_counts),'max byte idx',max(byte_counts))
print('flags in bytes >=311:',sorted([(k,hex(n),n//8,n%8) for k,(n,_) in ev.items() if n//8>=311],key=lambda t:t[2]))
