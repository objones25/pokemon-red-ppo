import json, sys, collections, ast, statistics as st, urllib.request
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
    r['reward']=asd(r['reward']); r['env']=asd(r['env']); r['policy']=asd(r['policy'])
    w=r['reward']
    if 'reward' in w and isinstance(w['reward'],dict): w=w['reward']
    r['w']=w
def rc(r):
    v=r['summary'].get('environment/stats/required_count'); return v if isinstance(v,(int,float)) else -1
win=[r for r in rows if isinstance(r['summary'].get(CH),(int,float)) and r['summary'][CH]>0]
print('sample reward shape', list(win[0]['reward'].keys())[:5])
print('winner sweep counts',collections.Counter(r['sweep'] for r in win))
lv=[r['w'].get('level') for r in win]; he=[r['w'].get('pokecenter_heal') for r in win]
print('level',min(lv),st.median(lv),max(lv)); print('heal',min(he),st.median(he),max(he))
print('champ==1',sum(r['summary'][CH]==1 for r in win))
for a in sorted(win[0]['env']):
    if a.startswith('auto_') or a in('skip_safari_zone','infinite_safari_steps','infinite_health','infinite_money','perfect_ivs','insert_saffron_guard_drinks','disable_wild_encounters','two_bit','init_state','required_tolerance'):
        print(a,collections.Counter(str(r['env'].get(a)) for r in win).most_common(4))
print('hidden sizes win',collections.Counter(str(r['policy'].get('rnn',{}).get('args')) for r in win))
print('all-run LSTM sizes',collections.Counter(str((r['policy'].get('rnn') or r['policy'].get('recurrent') or {}).get('args',{}).get('hidden_size')) for r in rows))
print('all-run policy sizes',collections.Counter(str((r['policy'].get('policy') or {}).get('hidden_size')) for r in rows))
k=[k for k in win[0]['summary'] if 'uptime' in k or 'agent_steps' in k or 'global_step' in k]; print(k)
for r in sorted(win,key=lambda r:r['summary'].get('performance/uptime',9e9))[:6]:
    print(r['name'],r['disp'],r['created'][:10],'uptime(s)',r['summary'].get('performance/uptime'),'steps',r['summary'].get('metric/agent_steps'),'lvl',round(r['w']['level'],3),'heal',round(r['w']['pokecenter_heal'],3), 'events',round(r['w'].get('event',0),2),'req_ev',round(r['w'].get('required_event',0),2),'badges',round(r['w'].get('badges',0),2),'explore',round(r['w'].get('exploration',0),4))
# level-weight ranges and low-level-weight runs
print()
X=[r for r in rows if isinstance(r['w'].get('level'),(int,float))]
print('runs with numeric level',len(X),'min',min(r['w']['level'] for r in X))
print('level==0',sum(r['w']['level']==0 for r in X))
print('level<0.2 by module',collections.Counter(r['reward_module'] for r in X if r['w']['level']<0.2))
for r in sorted([r for r in X if r['w']['level']<0.3],key=rc,reverse=True)[:12]:
    print(r['name'],r['created'][:10],r['state'],r['reward_module'][9:],r['sweep'],'lvl=%.4g'%r['w']['level'],'rc=%.1f'%rc(r),'champ',r['summary'].get(CH),'badge',r['summary'].get('environment/stats/badge'))
noleve=[r for r in rows if r['w'] and 'level' not in r['w']]
print('runs with reward block lacking level',len(noleve),collections.Counter(r['reward_module'] for r in noleve))
print('with empty reward',sum(1 for r in rows if not r['w']),collections.Counter(r['reward_module'] for r in rows if not r['w']).most_common(6))
# rank correlation inside each sweep
def spearman(a,b):
    def rk(v):
        s=sorted(range(len(v)),key=lambda i:v[i]); o=[0]*len(v)
        i=0
        while i<len(s):
            j=i
            while j+1<len(s) and v[s[j+1]]==v[s[i]]: j+=1
            for t in range(i,j+1): o[s[t]]=(i+j)/2
            i=j+1
        return o
    ra,rb=rk(a),rk(b)
    ma,mb=sum(ra)/len(ra),sum(rb)/len(rb)
    n=sum((x-ma)*(y-mb) for x,y in zip(ra,rb)); d=(sum((x-ma)**2 for x in ra)*sum((y-mb)**2 for y in rb))**.5
    return n/d if d else float('nan')
by=collections.defaultdict(list)
for r in X:
    if r['sweep'] and rc(r)>=0 and isinstance(r['w'].get('pokecenter_heal'),(int,float)): by[r['sweep']].append(r)
print('sweep n first mod  level_range heal_range rc_max rho(level,rc) rho(heal,rc)')
for s,v in sorted(by.items(),key=lambda kv:kv[1][0]['created']):
    if len(v)<8: continue
    L=[r['w']['level'] for r in v];H=[r['w']['pokecenter_heal'] for r in v];R=[rc(r) for r in v]
    print(s,len(v),v[0]['created'][:10],v[0]['reward_module'][9:30],'%.3g-%.3g'%(min(L),max(L)),'%.3g-%.3g'%(min(H),max(H)),'%.0f'%max(R),'%.2f'%spearman(L,R),'%.2f'%spearman(H,R), v[0]['env'].get('skip_safari_zone'),v[0]['env'].get('auto_use_cut'))
