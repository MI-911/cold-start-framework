#!/bin/bash
cd ..
docker build -f Dockerfile.interview -t mindreader/interview .
# docker run --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug

docker run --rm -d -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --models top-pop random naive-ppr-collab naive-mf
docker run --rm -d -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --models naive-ppr-joint naive-ppr-kg
docker run --rm -d -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --models naive-ppr-joint naive-ppr-kg
docker run --rm -d -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --models fmf lrmf
docker run --rm -d -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --models melu
