import json, sys, collections
rows=json.load(open(sys.argv[1]))
CH='environment/events/EVENT_BEAT_CHAMPION_RIVAL'
print(len(rows), rows[0]['created'], rows[-1]['created'])
print(collections.Counter(r['reward_module'] for r in rows).most_common())
print('rnn hidden',collections.Counter(str((r['rnn'] or {}).get('args') if isinstance(r['rnn'],dict) else r['rnn']) for r in rows).most_common(8))
print('policy',collections.Counter(str(r['policy']) for r in rows).most_common(5))
def rc(r):
    v=r['summary'].get('environment/stats/required_count'); return v if isinstance(v,(int,float)) else None
winners=[r for r in rows if isinstance(r['summary'].get(CH),(int,float)) and r['summary'][CH]>0]
print('winners (summary champion>0):',len(winners))
print(collections.Counter((r['reward_module'],r['created'][:7]) for r in winners).most_common())
A=['auto_flash','auto_teach_cut','auto_use_cut','auto_use_strength','auto_teach_strength','auto_use_surf','auto_teach_surf','auto_solve_strength_puzzles','auto_remove_all_nonuseful_items','auto_pokeflute','auto_next_elevator_floor','skip_safari_zone','infinite_money','infinite_health','perfect_ivs','insert_saffron_guard_drinks','disable_wild_encounters']
def row(r):
    w=r['reward'] or {}
    e=r['env'] or {}
    off=[a.replace('auto_','') for a in A if e.get(a) and a not in('infinite_money',)]
    return (r['created'][:16],r['name'],r['disp'],r['state'],r['sweep'],'champ=%.3f'%r['summary'].get(CH,0),'rc=%s'%rc(r),'lvl=%s'%w.get('level'),'heal=%s'%w.get('pokecenter_heal'),'hs=%s'%((r['rnn'] or {}).get('args',{}).get('hidden_size') if isinstance(r['rnn'],dict) else r['rnn']),'pol=%s'%(r['policy'] or {}).get('hidden_size') if isinstance(r['policy'],dict) else r['policy'],e.get('init_state'),e.get('max_steps'),off)
for r in sorted(winners,key=lambda r:r['created']): print(*row(r))
