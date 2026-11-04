import json,sys,urllib.request
def g(q,**v):
    req=urllib.request.Request("https://api.wandb.ai/graphql",data=json.dumps({"query":q,"variables":v}).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req,timeout=60))
print(g('query{entity(name:"thatguy11325"){projects(first:20){edges{node{name runCount access}}}}}'))
for n in sys.argv[1:]:
    print(n,g('query($n:String!){project(name:"pokemon",entityName:"thatguy11325"){run(name:$n){commit github state createdAt historyLineCount}}}',n=n))
