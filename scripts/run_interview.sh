#!/bin/bash
bash build_base.sh
cd ..
docker build -f Dockerfile.interview -t mindreader/interview .

EXPERIMENT="equal"
BASE=${PWD}

docker run -d --rm -v $BASE/data:/app/data -v $BASE/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-ppr-joint
docker run -d --rm -v $BASE/data:/app/data -v $BASE/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-ppr-joint --recommendable