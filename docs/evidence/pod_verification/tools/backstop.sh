#!/bin/bash
# Deletes this pod after $1 seconds using the pod-scoped key, with no key on disk. Set up for one pod only.
sleep "$1"
key=$(tr '\0' '\n' </proc/1/environ | awk -F= '$1=="RUNPOD_API_KEY"{print substr($0,length($1)+2)}')
id=$(tr '\0' '\n' </proc/1/environ | awk -F= '$1=="RUNPOD_POD_ID"{print substr($0,length($1)+2)}')
echo "$(date -u) backstop firing for $id" >> /workspace/podverify/backstop.log
curl -s -o /dev/null -w '%{http_code}\n' -X DELETE -H "Authorization: Bearer $key" "https://api.runpod.io/v2/pods/$id" >> /workspace/podverify/backstop.log
