#!/bin/bash
cd ..
docker build -f Dockerfile.interview -t mindreader/interview .
# docker run --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug

docker run -d -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --include top-pop random naive-ppr-collab naive-mf
docker run -d -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --include naive-ppr-joint naive-ppr-kg
docker run -d -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --include lrmf
docker run -d -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --include fmf
#docker run -d -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --models melu
