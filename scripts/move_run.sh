#!/bin/bash
cd ..
mkdir -p .runs/"$1"
mv results .runs/"$1"
mv data .runs/"$1"
mv sources .runs/"$1"
