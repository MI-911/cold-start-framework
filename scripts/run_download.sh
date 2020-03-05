#!/bin/bash
cd ..
docker build -f Dockerfile.download -t mindreader/downloader .
docker run --rm -v "${PWD}"/sources/mindreader:/app/sources/mindreader mindreader/downloader --output sources/mindreader
