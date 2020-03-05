#!/bin/bash
cd ..
docker build -f Dockerfile.base -t mindreader/base .
docker build -f Dockerfile.interview -t mindreader/interview .
docker run -d --rm mindreader/interview