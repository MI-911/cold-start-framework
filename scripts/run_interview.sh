#!/bin/bash
bash build_base.sh
cd ..
docker build -f Dockerfile.interview -t mindreader/interview .

EXPERIMENT="equal"
BASE=${PWD}

docker run -d -v $BASE/data:/app/data -v $BASE/results:/app/results mindreader/interview --upload --input data --output results --debug --experiments $EXPERIMENT --include greedy-ppr-linear-joint-learned
docker run -d -v $BASE/data:/app/data -v $BASE/results:/app/results mindreader/interview --upload --input data --output results --debug --experiments $EXPERIMENT --include greedy-ppr-linear-joint-learned --recommendable