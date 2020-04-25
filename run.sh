#!/bin/bash

for i in {0..3}
do
    for CONTEXT in 0 1
    do
        for ATTENTION in 0 1
        do
            CONTEXT=$CONTEXT ATTENTION=$ATTENTION python3 main.py
        done
    done
done