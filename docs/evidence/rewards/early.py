import json,time,urllib.request,collections
def g(q,**v):
    req=urllib.request.Request("https://api.wandb.ai/graphql",data=json.dumps({"query":q,"variables":v}).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req,timeout=120))
q='query($c:String){project(name:"pokemon",entityName:"thatguy11325"){runs(first:100,after:$c,order:"+created_at"){pageInfo{hasNextPage endCursor} edges{node{name displayName state createdAt commit config summaryMetrics}}}}}'
cur=None;rows=[]
for _ in range(3):
    d=g(q,c=cur)['data']['project']['runs']
    for e in d['edges']:
        n=e['node'];s=json.loads(n['summaryMetrics'] or '{}');c=json.loads(n['config'] or '{}')
        rm=c.get('reward_module',{}); rm=rm.get('value') if isinstance(rm,dict) else rm
        steps=s.get('performance/total_agent_steps') or s.get('Overview/agent_steps') or 0
        rows.append(dict(t=n['createdAt'][:16],name=n['displayName'],st=n['state'],mod=(rm or '')[9:],commit=(n['commit'] or '')[:7],steps=steps,lvl=s.get('stats/reward/level'),mxlvl=s.get('max_stats/reward/level'),badge=s.get('stats/stats/badge'),mxbadge=s.get('max_stats/stats/badge'),sumlv=s.get('stats/stats/levels_sum'),event=s.get('stats/stats/event') ))
    if not d['pageInfo']['hasNextPage']: break
    cur=d['pageInfo']['endCursor']; time.sleep(.3)
rows=[r for r in rows if r['t']<'2024-04-20']
print(len(rows),rows[0]['t'],rows[-1]['t'])
# group by module and day, only runs with steps > 20M
for r in rows:
    if r['steps'] and r['steps']>2e7:
        print(r['t'],r['name'][:22],r['st'][:4],r['mod'][:26],r['commit'],'steps=%.0fM'%(r['steps']/1e6),'lvlrew',r['lvl'],'mx',r['mxlvl'],'badge',None if r['badge'] is None else round(r['badge'],2),'mxb',r['mxbadge'],'sum',None if r['sumlv'] is None else round(r['sumlv'],1))
