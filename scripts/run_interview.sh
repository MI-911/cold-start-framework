#!/bin/bash
cd ..
docker build -f Dockerfile.base -t mindreader/base .
docker build -f Dockerfile.interview -t mindreader/interview .
docker run --rm -v "${PWD}"/data:/app/data mindreader/interview --input data
