#!/bin/bash
RESULTS="/media/calc_2/scone_results/live/lalitha"
PROJ_DIR="/home/nadinebadie/lalitha/depRL-sconegym"
NUM_EPISODES=5

RUNS=(
    "walk_depmpo_h0918_onlyVelRew"
    "walk_mpo_h0918_onlyVelRew"
    "run_depmpo_h0918_onlyVelRew"
    "run_mpo_h0918_onlyVelRew"
)

cd $PROJ_DIR
for run in "${RUNS[@]}"; do
    path="${RESULTS}/${run}/$(ls ${RESULTS}/${run}/)"
    echo "Playing: $path"
    python -m deprl.play --path "$path" --num_episodes $NUM_EPISODES
done
