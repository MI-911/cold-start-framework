#!/bin/bash
bash build_base.sh
cd ..
docker build -f Dockerfile.interview -t mindreader/interview .

EXPERIMENT="equal"
BASE="/home/anders/Code/cold-start-framework/debug"

#docker run -d -v $BASE/data:/app/data -v $BASE/results:/app/results mindreader/interview --upload --input data --output results --debug --experiments $EXPERIMENT --include greedy-adaptive-mf
# --recommendable
docker run -v $BASE/data:/app/data -v $BASE/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-adaptive-ppr-joint --limit decade
#docker run -d -v $BASE/data:/app/data -v $BASE/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-adaptive-ppr-joint --limit category
#docker run -d -v $BASE/data:/app/data -v $BASE/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-adaptive-ppr-joint --limit company
#docker run -d -v $BASE/data:/app/data -v $BASE/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-adaptive-ppr-joint --limit person
# docker run -d -v $BASE/data:/app/data -v $BASE/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-adaptive-ppr-joint --limit decade category company movie person