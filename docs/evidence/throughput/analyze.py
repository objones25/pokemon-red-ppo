import json,sys,collections
R=[json.loads(l) for l in open('results/runs.jsonl')]
G=[json.loads(l) for l in open('results/gate.jsonl')]
print(len(R),len(G))
for r,g in zip(R,G):
    assert r['label']==g['name'] and abs(r['sps']-g['sps'])<1e-6
    f=r.get('frac',{}); m=r.get('ms_per_lockstep',{}); u=r.get('us_per_env_step',{}); e=r['mem']
    fs=' '.join(f"{k}={100*v:.1f}" for k,v in f.items())
    print(f"{r['label']:13s} rd{g['round']:<2d} sps={r['sps']:6.0f} wall={r['wall']:.0f} cyc={r['cycles']} bg={g['bg_mean']:.2f}/{g['bg_max']:.1f} own={g['own_cores']:.1f} gpu0={g['gpu_util_pre']} la={g['load_pre'][0]:.1f}->{g['load_post'][0]:.1f} {'FLAG' if g['contaminated'] else 'ok'} rst={r.get('resets')}\n    {fs}\n    lock ms env={m.get('env',0):.2f} emu={m.get('emu',0):.2f} obs={m.get('obs',0):.3f} rew={m.get('rew',0):.3f} sync={m.get('sync_ipc',0):.2f} | emu/step={u.get('emu',0):.0f}us | rss={e['rss_mb_mean']:.0f} uss={e['uss_mb_mean']}")
print(G[-1]['thermal']); print(set(g['thermal'][-60:] for g in G))
print([ (g['name'],g['top_others_pre']) for g in G if g['top_others_pre']][:6])

print('\n--- per config, clean=bg<=1.0; ms per lockstep and share')
by=collections.defaultdict(list)
for r,g in zip(R,G): by[r['label']].append((r,g))
for name,x in by.items():
    allv=[r['sps'] for r,g in x]; cl=[r['sps'] for r,g in x if not g['contaminated']]
    use=[r for r,g in x if not g['contaminated']] or [r for r,g in x]
    f=collections.Counter(); m=collections.Counter()
    for r in use:
        for k,v in r['frac'].items(): f[k]+=v/len(use)
        for k,v in r['ms_per_lockstep'].items(): m[k]+=v/len(use)
    S=sum(r['sps'] for r in use)/len(use); N=use[0]['N']; wl=N/S*1e3
    infer=f['xfer']+f['fwd']+f['tocpu']+f['buf']+f['collect_rest']
    print(f"{name:13s} all={[round(v) for v in allv]} clean_mean={sum(cl)/len(cl) if cl else float('nan'):.0f} wall/lock={wl:.2f}ms | emu={100*m['emu']/wl:.1f}% obs+rew={100*(m['obs']+m['rew'])/wl:.1f}% sync/ipc={100*m['sync_ipc']/wl:.1f}% env={100*f['env']:.1f}% | rollout-policy={100*infer:.1f}% update={100*f['update']:.1f}% gae={100*f['gae']:.1f}% | per-sample ms: emu={m['emu']/N:.3f} sync={m['sync_ipc']/N:.3f} pol={infer*wl/N:.3f} upd={f['update']*wl/N:.3f} tot={wl/N:.3f}")
print('bg range',min(g['bg_mean'] for g in G),max(g['bg_mean'] for g in G),'flagged',sum(bool(g['contaminated']) for g in G),'of',len(G))
