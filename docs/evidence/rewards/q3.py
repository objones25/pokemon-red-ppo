import json,sys,time,urllib.request
def g(q,**v):
    req=urllib.request.Request("https://api.wandb.ai/graphql",data=json.dumps({"query":q,"variables":v}).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req,timeout=120))
q='query($c:String){project(name:"pufferlib",entityName:"thatguy11325"){runs(first:100,after:$c,order:"+created_at"){pageInfo{hasNextPage endCursor} edges{node{name displayName state createdAt heartbeatAt commit summaryMetrics}}}}}'
cur=None;rows=[]
while True:
    d=g(q,c=cur)['data']['project']['runs']
    for e in d['edges']:
        n=e['node'];s=json.loads(n['summaryMetrics'] or '{}')
        rows.append((n['createdAt'],n['heartbeatAt'],n['name'],n['displayName'],n['state'],n['commit'],s.get('environment/badge'),s.get('environment/event'),s.get('environment/level'),s.get('environment/levels_sum'),s.get('environment/stats/badge'),s.get('environment/max_map_progress'),s.get('environment/reward'),s.get('environment/explore'),s.get('performance/uptime'),[k for k in s if 'badge' in k.lower() or 'level' in k.lower()][:6]))
    if not d['pageInfo']['hasNextPage']: break
    cur=d['pageInfo']['endCursor']; time.sleep(.3)
print(len(rows),rows[0][0],rows[-1][0])
for r in rows:
    if '2024-02-08'<r[0]<'2024-02-16': print(*r[:6],'|',*[('%.2f'%x if isinstance(x,float) else x) for x in r[6:15]],r[15][:6] if r[0]<'2024-02-09' else '')
