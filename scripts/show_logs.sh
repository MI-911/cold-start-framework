#!/bin/bash

function _exit {
  kill $(jobs -p)
}

trap _exit EXIT

for name in $(docker ps --format "{{.Names}}"); do
  eval "docker logs -f --tail=5 \"$name\" | sed -e \"s/^/[-- $name --] /\" &";
done

wait
