#!/bin/bash
cd ..
docker build -f Dockerfile.interview -t mindreader/interview .

EXPERIMENT="default"

#docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include random top-pop
#docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include naive-mf
#docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include naive-ppr-collab
#docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include naive-ppr-joint
#docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include naive-ppr-kg
#docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-ppr-kg
#docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-ppr-kg-rec
#docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-ppr-joint
#docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-ppr-joint-rec
#docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-ppr-collab
#docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-ppr-collab-rec
#docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include lrmf
#docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include fmf
docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-knn
docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include melu
docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-ppr-linear-joint
docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-ppr-linear-kg
docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include greedy-ppr-linear-collab
docker run -d --rm -v "${PWD}"/data:/app/data -v "${PWD}"/results:/app/results mindreader/interview --input data --output results --debug --experiments $EXPERIMENT --include random