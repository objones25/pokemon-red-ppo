import json, sys, time, urllib.request
out = sys.argv[1]
q = '''query R($c:String){project(name:"pokemon",entityName:"thatguy11325"){runs(first:100,after:$c,order:"+created_at"){pageInfo{hasNextPage endCursor} edges{node{name displayName state createdAt heartbeatAt sweepName summaryMetrics config}}}}}'''
cur=None; rows=[]
while True:
    req=urllib.request.Request("https://api.wandb.ai/graphql",data=json.dumps({"query":q,"variables":{"c":cur}}).encode(),headers={"Content-Type":"application/json"})
    d=json.load(urllib.request.urlopen(req,timeout=120))
    if "errors" in d: print(d["errors"]); break
    r=d["data"]["project"]["runs"]
    for e in r["edges"]:
        n=e["node"]; s=json.loads(n["summaryMetrics"] or "{}"); c=json.loads(n["config"] or "{}")
        def v(x):
            return x.get("value") if isinstance(x,dict) and "value" in x else x
        keep={k:s.get(k) for k in s if any(w in k for w in ("EVENT_BEAT_","required_count","stats/badge","reward/level","reward/pokecenter_heal","stats/levels_sum","stats/level","pokecenter_heal","performance/uptime","agent_steps","global_step","stats/deaths","stats/blackout"))}
        rows.append(dict(name=n["name"],disp=n["displayName"],state=n["state"],created=n["createdAt"],hb=n["heartbeatAt"],sweep=n["sweepName"],summary=keep,reward=v(c.get("reward")),reward_module=v(c.get("reward_module")),env=v(c.get("env")),wrappers=v(c.get("wrappers")),rnn=v(c.get("rnn")),policy=v(c.get("policy")),train_keys={k:v(c[k]) for k in c if k in ("cleanrl",)}))
    print(len(rows),flush=True)
    if not r["pageInfo"]["hasNextPage"]: break
    cur=r["pageInfo"]["endCursor"]; time.sleep(0.3)
json.dump(rows,open(out,"w"))
