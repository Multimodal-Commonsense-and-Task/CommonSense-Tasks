#!/bin/bash

# Wait for 1 hour
sleep 5h

# Run with different GPUs
bash run.sh 0 &
bash run.sh 1 &
bash run.sh 2 &
bash run.sh 3 &
