#!/bin/bash
cd ..
docker build -f Dockerfile.interview -t mindreader/interview .

docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --experiments default --include naive-ppr-joint
docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --experiments default --include naive-ppr-kg
docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --experiments default --include naive-ppr-collab
docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --experiments default --include lrmf
