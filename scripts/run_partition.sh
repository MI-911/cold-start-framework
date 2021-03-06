#!/bin/bash
cd ..
docker build -f Dockerfile.partition -t mindreader/partitioner .
docker run --rm -v "${PWD}"/data:/app/data -v "${PWD}"/sources/mindreader:/app/source mindreader/partitioner --input source --output data
