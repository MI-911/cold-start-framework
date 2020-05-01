#!/bin/bash
for MODEL in "$@"
do
    sh model_runner.sh $MODEL
done
