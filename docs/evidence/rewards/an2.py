import json, sys, collections, ast
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
    if 'policy' in r['reward'] : pass
A=['auto_teach_cut','auto_use_cut','auto_use_strength','auto_teach_strength','auto_use_surf','auto_teach_surf','auto_solve_strength_puzzles','auto_remove_all_nonuseful_items','auto_pokeflute','auto_next_elevator_floor','skip_safari_zone','infinite_safari_steps','infinite_health','perfect_ivs','insert_saffron_guard_drinks','disable_wild_encounters']
def rc(r):
    v=r['summary'].get('environment/stats/required_count'); return v if isinstance(v,(int,float)) else -1
def fmt(r):
    w=r['reward'];e=r['env']
    return ' '.join(map(str,(r['created'][:16],r['name'],r['disp'],r['state'],r['sweep'],'champ=%s'%r['summary'].get(CH),'rc=%.1f'%rc(r),'lvl=%s'%w.get('level'),'heal=%s'%w.get('pokecenter_heal'),'hs=%s'%((r['policy'].get('rnn') or {}).get('args') or {}).get('hidden_size'),e.get('init_state'),e.get('max_steps'),'|'.join(a.replace('auto_','') for a in A if e.get(a)))))
win=[r for r in rows if isinstance(r['summary'].get(CH),(int,float)) and r['summary'][CH]>0]
print('WINNERS by summary',len(win))
print('level weights',collections.Counter(round(r['reward'].get('level',-1),2) for r in win if isinstance(r['reward'].get('level',-1),(int,float))).most_common(12))
print('heal weights',collections.Counter(str(r['reward'].get('pokecenter_heal'))[:5] for r in win).most_common(12))
for r in sorted(win,key=lambda r:(r['summary'][CH]),reverse=True)[:12]: print(fmt(r))
print()
print('TOP rc by module')
mods=collections.defaultdict(list)
for r in rows: mods[r['reward_module']].append(r)
for m,v in mods.items():
    v=sorted(v,key=rc,reverse=True); 
    print('==',m,len(v), 'rc>=70:',sum(rc(x)>=70 for x in v))
    for r in v[:3]: print(fmt(r)[:300])
print()
print('FieldMoves after 2025-02-26')
late=[r for r in rows if r['created']>='2025-02-20' and r['reward_module'] and 'FieldMoves' in r['reward_module']]
print(len(late))
for r in sorted(late,key=rc,reverse=True)[:25]: print(fmt(r)[:320])
print()
print('level absent/zero')
z=[r for r in rows if r['reward'] and (r['reward'].get('level') in (None,0,0.0))]
print(collections.Counter(r['reward_module'] for r in z).most_common())
for r in sorted(z,key=rc,reverse=True)[:10]: print(fmt(r)[:320])
