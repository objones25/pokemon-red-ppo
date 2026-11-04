import json, sys, collections, ast, time, urllib.request
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
    r['reward']=asd(r['reward']); r['env']=asd(r['env'])
    r['w']=r['reward'].get('reward') if isinstance(r['reward'].get('reward'),dict) else r['reward']
def rc(r):
    v=r['summary'].get('environment/stats/required_count'); return v if isinstance(v,(int,float)) else -1
win=[r for r in rows if isinstance(r['summary'].get(CH),(int,float)) and r['summary'][CH]>0]
print('Winner exceptions:')
for r in win:
    if not r['env'] or r['env'].get('skip_safari_zone') is not True:
        print(r['name'],r['disp'],r['created'][:16],r['state'],r['sweep'],r['reward_module'],'champ',r['summary'][CH],'rc',rc(r),r['w'].get('level'),r['w'].get('pokecenter_heal'),{k:r['env'].get(k) for k in('skip_safari_zone','auto_flash','disable_wild_encounters','insert_saffron_guard_drinks','auto_next_elevator_floor','perfect_ivs','init_state')})
print('winner wrappers',collections.Counter(str(r['wrappers'])[:90] for r in win).most_common(4))
t=sorted(win,key=lambda r:r['summary'].get('performance/uptime',9e9))[0]
print('fast winner cleanrl',{k:t['train_keys'].get('cleanrl',{}).get(k) for k in('gamma','learning_rate','ent_coef','bptt_horizon','num_envs','batch_size','minibatch_size','update_epochs','clip_coef')} if isinstance(t['train_keys'].get('cleanrl'),dict) else str(t['train_keys'])[:300])
print('uptime hours range winners', min(r['summary'].get('performance/uptime',9e9) for r in win)/3600, max(r['summary'].get('performance/uptime',0) for r in win)/3600)
print('reward modules & max rc & champ count by period')
pc=[r for r in rows if isinstance(r['w'].get('pokecenter_heal'),(int,float))]
print('heal 0.75 runs:',len([r for r in pc if abs(r['w']['pokecenter_heal']-.75)<1e-6]), 'max rc', max([rc(r) for r in pc if abs(r['w']['pokecenter_heal']-.75)<1e-6] or [-1]))
for lo,hi in ((0,.3),(.3,.6),(.6,.75),(.75,.76),(.76,2)):
    v=[r for r in pc if lo<=r['w']['pokecenter_heal']<hi]; print('heal',lo,hi,len(v),'maxrc',max([rc(r) for r in v] or [-1]),'champ>0',sum(1 for r in v if (r['summary'].get(CH) or 0)>0))
print('late field moves w/ scripts on (Mar 2025+)')
late=[r for r in rows if r['created']>='2025-02-27' and r['reward_module'] and 'FieldMoves' in r['reward_module']]
print(len(late),collections.Counter(str((r['env'].get('auto_use_surf'),r['env'].get('auto_pokeflute'))) for r in late).most_common(4),'envs empty',sum(not r['env'] for r in late))
for r in sorted(late,key=rc,reverse=True)[:3]: print(r['name'],r['disp'],r['created'][:10],rc(r),r['w'].get('level'),r['w'].get('pokecenter_heal'),str(r['env'])[:80])
print('rock tunnel runs (hard cap window Apr 16-18 2024)')
for r in rows:
    if r['reward_module'] and 'RockTunnel' in r['reward_module'] or ('2024-04-15'<r['created']<'2024-04-20'):
        print(r['name'],r['disp'],r['created'][:16],r['state'],r['reward_module'],'uptime',r['summary'].get('performance/uptime'),{k[-22:]:v for k,v in r['summary'].items() if 'EVENT_BEAT_' in k and ('BROCK' in k or 'MISTY' in k or 'SURGE' in k or 'ERIKA' in k)})
