#!/bin/bash
bash build_base.sh
cd ..
docker build -f Dockerfile.interview -t mindreader/interview .

EXPERIMENT="equal"
BASE=${PWD}

docker run -d -v $BASE/data:/app/data -v $BASE/additional_results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-adaptive-ppr-joint
docker run -d -v $BASE/data:/app/data -v $BASE/additional_results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-adaptive-ppr-joint --recommendable
docker run -d -v $BASE/data:/app/data -v $BASE/additional_results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-adaptive-ppr-joint --limit decade
docker run -d -v $BASE/data:/app/data -v $BASE/additional_results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-adaptive-ppr-joint --limit category
docker run -d -v $BASE/data:/app/data -v $BASE/additional_results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-adaptive-ppr-joint --limit company
docker run -d -v $BASE/data:/app/data -v $BASE/additional_results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-adaptive-ppr-joint --limit person
docker run -d -v $BASE/data:/app/data -v $BASE/additional_results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-adaptive-ppr-joint --limit decade category company movie person