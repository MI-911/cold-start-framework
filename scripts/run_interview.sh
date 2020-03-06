#!/bin/bash
cd ..
docker build -f Dockerfile.interview -t mindreader/interview .
docker run --rm -v "${PWD}"/data:/app/data -v "${PWD}"/.results:/app/results mindreader/interview --input data --include naive
