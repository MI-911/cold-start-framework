#!/bin/bash
cd ..
docker build -f Dockerfile.interview -t mindreader/interview .

EXPERIMENT="greedy_test"

docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --experiments $EXPERIMENT --include naive-ppr-collab naive-ppr-kg
docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --experiments $EXPERIMENT --include greedy-ppr-collab greedy-ppr-kg
docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --experiments $EXPERIMENT --include greedy-ppr-collab-limited greedy-ppr-kg-limited
docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --experiments $EXPERIMENT --include naive-ppr-joint
docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --experiments $EXPERIMENT --include greedy-ppr-joint-limited
docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --experiments $EXPERIMENT --include greedy-ppr-joint
docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --experiments $EXPERIMENT --include top-pop random