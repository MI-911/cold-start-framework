#!/bin/bash
bash build_base.sh
cd ..
docker build -f Dockerfile.interview -t mindreader/interview .

EXPERIMENT="separation"
BASE="/home/anders/Code/cold-start-framework/debug"

docker run -d -v $BASE/data:/app/data -v $BASE/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-adaptive-mf
docker run -d -v $BASE/data:/app/data -v $BASE/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-adaptive-mf --recommendable
docker run -d -v $BASE/data:/app/data -v $BASE/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-adaptive-knn
docker run -d -v $BASE/data:/app/data -v $BASE/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-adaptive-knn --recommendable
