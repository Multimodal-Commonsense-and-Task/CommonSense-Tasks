#!/bin/bash
#SBATCH -p dell_cpu
#SBATCH -q cpu_qos
#SBATCH -c 32

tar -cf generated_datasets.tar generated_datasets/
