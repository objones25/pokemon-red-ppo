import json, sys, urllib.request
DCS = "AP-JP-1 CA-MTL-3 CA-MTL-4 EU-CZ-1 EU-NL-1 EU-RO-1 EUR-IS-1 EUR-IS-3 EUR-IS-4 EUR-NO-1 US-CO-1 US-IL-1 US-NC-1 US-NE-1 US-TX-3 US-WA-1 AP-IN-1 EU-FR-1 EUR-NO-2 US-CA-2".split()
parts = []
for g, gid in (("sxm", "NVIDIA A100-SXM4-80GB"), ("pcie", "NVIDIA A100 80GB PCIe")):
    for i, dc in enumerate(DCS):
        parts.append('%s_%d: gpuTypes(input:{id:"%s"}) { lowestPrice(input:{gpuCount:1, secureCloud:true, dataCenterId:"%s"}) { stockStatus uninterruptablePrice minVcpu minMemory } }' % (g, i, gid, dc))
q = "query {" + " ".join(parts) + "}"
req = urllib.request.Request("https://api.runpod.io/graphql", json.dumps({"query": q}).encode(), {"content-type": "application/json", "user-agent": "curl/8"})
data = json.load(urllib.request.urlopen(req, timeout=30))
if "errors" in data: print("ERR", str(data["errors"])[:300])
found = []
for k, v in (data.get("data") or {}).items():
    lp = v[0]["lowestPrice"]
    if lp and lp.get("stockStatus"):
        g, i = k.split("_"); found.append((g, DCS[int(i)], lp["stockStatus"], lp["uninterruptablePrice"], lp.get("minVcpu")))
q2 = "query {" + " ".join('c%d: gpuTypes(input:{id:\"%s\"}) { lowestPrice(input:{gpuCount:1, secureCloud:false}) { stockStatus uninterruptablePrice minVcpu } }' % (i, g) for i, g in enumerate(("NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"))).replace('\\','') + "}"
req = urllib.request.Request("https://api.runpod.io/graphql", json.dumps({"query": q2}).encode(), {"content-type": "application/json", "user-agent": "curl/8"})
d2 = json.load(urllib.request.urlopen(req, timeout=30)).get("data") or {}
for k, v in d2.items():
    lp = v[0]["lowestPrice"]
    if lp and lp.get("stockStatus"):
        found.append(("community-" + k, lp["stockStatus"], lp["uninterruptablePrice"], lp.get("minVcpu")))
print(found)
