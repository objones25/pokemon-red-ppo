import json,sys,time,urllib.request,collections
def g(q,**v):
    req=urllib.request.Request("https://api.wandb.ai/graphql",data=json.dumps({"query":q,"variables":v}).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req,timeout=120))
Q='query($n:String!){project(name:"pokemon",entityName:"thatguy11325"){run(name:$n){summaryMetrics}}}'
s=json.loads(g(Q,n='e4p7tuad')['data']['project']['run']['summaryMetrics'])
print([k for k in s if 'level' in k.lower() or k in('environment/stats/deaths','environment/stats/blackout_count')])
print({k:s[k] for k in ('environment/reward/level','environment/stats/badge','environment/stats/required_count','environment/stats/caught_pokemon','environment/stats/deaths','environment/stats/blackout_count') if k in s})
H='query($n:String!,$s:[JSONString!]!){project(name:"pokemon",entityName:"thatguy11325"){run(name:$n){sampledHistory(specs:$s)}}}'
W={'e4p7tuad':None,'08y1j69u':1.036,'roue3btn':0.998,'elevator':1.05,'daily':None}
for n,lw in (('e4p7tuad',None),('08y1j69u',1.036),('roue3btn',0.998),('elevator',1.05),('n7ndo789',None),('l9xkwp19',1.0989),('mads1p13',1.0989)):
    d=g(H,n=n,s=[json.dumps({"keys":['environment/stats/badge','environment/reward/level'],"samples":800}),json.dumps({"keys":['environment/stats/required_count'],"samples":800}),json.dumps({"keys":['environment/events/EVENT_BEAT_CHAMPION_RIVAL'],"samples":800})])
    if 'errors' in d: print(n,d['errors']); continue
    h,rc,ch=d['data']['project']['run']['sampledHistory']
    by=collections.defaultdict(list)
    for p in h: by[int(p['environment/stats/badge']+.5)].append(p['environment/reward/level'])
    print(n,'pts',len(h),'maxrc',max([p['environment/stats/required_count'] for p in rc] or [-1]),'maxchamp',max([p['environment/events/EVENT_BEAT_CHAMPION_RIVAL'] for p in ch] or [-1]),'maxbadge',max([p['environment/stats/badge'] for p in h] or [-1]))
    print('   reward/level by rounded mean badge (n,med,max):',{b:(len(v),round(sorted(v)[len(v)//2],1),round(max(v),1)) for b,v in sorted(by.items())})
    time.sleep(.3)
for n in ('lxe8wfhi','si8fwk9d','nnpmld8x','zm08nszd'):
    s=json.loads(g(Q,n=n)['data']['project']['run']['summaryMetrics'] or '{}')
    print(n,len(s),{k:(round(s[k],2) if isinstance(s[k],float) else s[k]) for k in s if any(w in k for w in('badge','level','performance/uptime','agent_steps'))})
