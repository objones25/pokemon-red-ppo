import json, sys, urllib.request
out = sys.argv[1]; order = sys.argv[2]; n = int(sys.argv[3])
q = '''query R($p:String!,$e:String!,$o:String,$n:Int){project(name:$p,entityName:$e){runs(first:$n,order:$o){edges{node{name displayName state createdAt group sweepName summaryMetrics config}}}}}'''
req = urllib.request.Request("https://api.wandb.ai/graphql", data=json.dumps({"query":q,"variables":{"p":"pokemon","e":"thatguy11325","o":order,"n":n}}).encode(), headers={"Content-Type":"application/json"})
d = json.load(urllib.request.urlopen(req, timeout=60))
json.dump(d, open(out,"w"))
if "errors" in d: print(d["errors"]); sys.exit()
for e in d["data"]["project"]["runs"]["edges"]:
    r = e["node"]; s = json.loads(r["summaryMetrics"] or "{}")
    print(r["name"], r["displayName"], r["state"], r["createdAt"], r["sweepName"], s.get("environment/stats/required_count"), s.get("environment/stats/badges"), s.get("global_step") or s.get("agent_steps"), [k for k in s if "CHAMPION" in k][:2])
