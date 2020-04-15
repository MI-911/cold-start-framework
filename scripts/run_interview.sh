#!/bin/bash
cd ..
docker build -f Dockerfile.interview -t mindreader/interview .
# docker run --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug

docker run -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --include random
# docker run -d -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --include naive-ppr-joint naive-ppr-kg
# docker run -d -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --include lrmf
# docker run -d -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --include fmf
#docker run -d -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --debug --models melu
