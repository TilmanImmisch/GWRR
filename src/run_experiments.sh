#!/bin/bash

# Script to reproduce results

#Environments to test
envs=("InvertedPendulum-v2" "Walker2d-v2" "HalfCheetah-v2" "Reacher-v2")

for environment in "${envs[@]}" ; do
    #Baseline
    python3 main.py \
    --env $environment \
    --ray_tune True \
    --replay_memory "original" \
    --batch_name "Baseline" \
    --num_ray_runs 10

    #Modulation at
    python3 main.py \
    --env $environment \
    --ray_tune True \
    --run_schedule 'modulate_at' \
    --batch_name "Modulate_at" \
    --num_ray_runs 10

    #Modulation ht
    python3 main.py \
    --env $environment \
    --ray_tune True \
    --run_schedule 'modulate_ht' \
    --batch_name "Modulate_ht" \
    --num_ray_runs 10
done
