import json,sys,ast,time,urllib.request,collections
rows=json.load(open(sys.argv[1]))
CH='environment/events/EVENT_BEAT_CHAMPION_RIVAL'
def asd(x):
    if isinstance(x,dict): return x
    if isinstance(x,str):
        try: return ast.literal_eval(x)
        except Exception:
            try: return json.loads(x)
            except Exception: return {}
    return {}
for r in rows:
    r['env']=asd(r['env']); rw=asd(r['reward']); r['w']=rw.get('reward') if isinstance(rw.get('reward'),dict) else rw; r['tr']=asd(asd(r['train_keys']).get('cleanrl'))
win=[r for r in rows if isinstance(r['summary'].get(CH),(int,float)) and r['summary'][CH]>0]
for k in ('auto_flash','auto_next_elevator_floor','insert_saffron_guard_drinks','disable_wild_encounters','skip_safari_zone'):
    print(k,'off in',[r['disp'] for r in win if r['env'] and not r['env'].get(k)])
def g(q,**v):
    req=urllib.request.Request("https://api.wandb.ai/graphql",data=json.dumps({"query":q,"variables":v}).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req,timeout=120))
# full summary keys of one winner containing level/heal
Q='query($n:String!){project(name:"pokemon",entityName:"thatguy11325"){run(name:$n){summaryMetrics config}}}'
d=g(Q,n='e4p7tuad')['data']['project']['run']; s=json.loads(d['summaryMetrics'])
print({k:s[k] for k in s if any(w in k.lower() for w in('level','heal','pokecenter','death','blackout','reward/event','reward/badges','reward/exploration','reward/required','reward_sum','caught','party'))})
print(sorted(k for k in s if k.startswith('environment/stats'))[:50])
print('winner summary level metric', sorted((r['disp'],round(r['summary'].get('environment/reward/level',-1),1),round(r['summary'].get('environment/reward/pokecenter_heal',-1),3)) for r in win)[:6])
c=win[0]['tr']; print({k:c.get(k) for k in('gamma','learning_rate','ent_coef','gae_lambda','vf_coef','clip_coef','bptt_horizon','num_envs','num_workers','batch_size','minibatch_size','update_epochs','total_timesteps','seed')})
print('seeds', collections.Counter(r['tr'].get('seed') for r in win), 'total_timesteps range',min(r['tr'].get('total_timesteps',9e18) for r in win),max(r['tr'].get('total_timesteps',0) for r in win))
print('ent_coef',min(r['tr'].get('ent_coef',9) for r in win),max(r['tr'].get('ent_coef',0) for r in win))
json.dump({},open('/dev/null','w'))
H='query($n:String!,$s:[JSONString!]!){project(name:"pokemon",entityName:"thatguy11325"){run(name:$n){sampledHistory(specs:$s)}}}'
keys=['environment/stats/levels_sum','environment/stats/badge','environment/stats/required_count',CH,'environment/reward/level','Overview/agent_steps']
for n in ('e4p7tuad','08y1j69u','roue3btn','n7ndo789','l9xkwp19','mads1p13'):
    spec=json.dumps({"keys":keys,"samples":400})
    d=g(H,n=n,s=[spec])
    h=d['data']['project']['run']['sampledHistory'][0] if 'data' in d else d
    print(n,len(h))
    best=collections.defaultdict(list)
    for p in h:
        b=p.get('environment/stats/badge')
        if b is None: continue
        best[round(b)].append(p.get('environment/stats/levels_sum') or 0)
    print('  max champ',max((p.get(CH) or 0) for p in h),'max rc',max((p.get('environment/stats/required_count') or 0) for p in h),'max badge',max((p.get('environment/stats/badge') or 0) for p in h),'max levels_sum',max((p.get('environment/stats/levels_sum') or 0) for p in h))
    print('  by rounded badge: (n,min,median,max) levels_sum', {b:(len(v),round(min(v)),round(sorted(v)[len(v)//2]),round(max(v))) for b,v in sorted(best.items())})
    time.sleep(.3)
for n in ('lxe8wfhi','si8fwk9d','nnpmld8x','zm08nszd'):
    d=g(Q,n=n)['data']['project']['run']; s=json.loads(d['summaryMetrics'] or '{}')
    print(n,len(s),{k:(round(s[k],2) if isinstance(s[k],float) else s[k]) for k in s if any(w in k for w in('badge','level','event','performance/uptime','agent_steps','explore'))})
