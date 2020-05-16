#!/bin/bash

for i in {0..2}; do
    for CONTEXT in 0 1; do
        for ATTENTION in 0 1; do
            LOAD_CHECKPOINT=0 CONTEXT=$CONTEXT ATTENTION=$ATTENTION python3 src/train.py
            CONTEXT=$CONTEXT ATTENTION=$ATTENTION python3 src/inference.py
            python3 src/evaluate.py
        done
    done
done
