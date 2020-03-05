#!/bin/bash
cd ..
docker build -f Dockerfile.base -t mindreader/base .
docker build -f Dockerfile.partition -t mindreader/partitioner .
docker run -it --rm -v "${PWD}"/data:/app/data -v "${PWD}"/sources/mindreader:/app/source mindreader/partitioner --input source --output data
