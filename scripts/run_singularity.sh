#!/bin/bash
lstkey=""
EXPERIMENTS=""
MODELS=""
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -e|--experiment)
    EXPERIMENTS="$2"
    lstkey=$key
    shift # past argument
    shift # past value
    ;;
    -i|--include)
    MODELS="$2"
    lstkey=$key
    shift # past argument
    shift # past value
    ;;
    *)    # unknown optionr
    if [ ${#lstkey} -ge 1 ];
    then
      case $lstkey in
        -e|--experiment)
         EXPERIMENTS="$EXPERIMENTS $key"
        ;;
        -i|--include)
        MODELS="$MODELS $key"
        ;;
      esac
    fi
    shift # past argument
    ;;
esac
done

if [ ${#MODELS} -lt 1 ]; then
    echo "ERROR: Must contain at least one model name.";
    echo "-e OR --experiments to define experiments.";
    echo "-i OR --include to define models"
    exit 1
fi

if [ ${#EXPERIMENTS} -eq 0 ];
then
  ECMD=""
  NAME=""
else
  ECMD="--experiments $EXPERIMENTS"
  NAME=""
  for experiment in $EXPERIMENTS
  do
    NAME="$NAME-$experiment"
  done
fi

for model in $MODELS
do
  echo "singularity instance start --nv -B ..:/app ../coldstart4.sigm $model$NAME interview.py --include $model $ECMD"
  singularity instance start --nv -B ..:/app ../coldstart.sigm "$model$NAME" interview.py --include $model $ECMD
  echo "sleep 1m"
  sleep 1m
done
