#!/bin/bash
bash build_base.sh
cd ..
docker build -f Dockerfile.interview -t mindreader/interview .

EXPERIMENT="equal"
BASE="/home/anders/Code/cold-start-framework/debug"

docker run -d --rm -v $BASE/data:/app/data -v $BASE/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-adaptive-ppr-joint --splits split_1 split_0
docker run -d --rm -v $BASE/data:/app/data -v $BASE/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-adaptive-ppr-joint --splits split_2 split_3
docker run -d --rm -v $BASE/data:/app/data -v $BASE/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-adaptive-ppr-joint --splits split_1 split_0 --recommendable
docker run -d --rm -v $BASE/data:/app/data -v $BASE/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-adaptive-ppr-joint --splits split_2 split_3 --recommendable