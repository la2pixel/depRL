#!/bin/bash
CONDA_ENV="lalitha"
PROJ_DIR="/home/nadinebadie/lalitha/depRL-sconegym"
BASE="experiments/hyfydy/baseline_runs/h2190"

NAMES=(
    "bl_walk_depmpo_h2190_velrew"   "bl_walk_mpo_h2190_velrew"
    "bl_run_depmpo_h2190_velrew"    "bl_run_mpo_h2190_velrew"
    "bl_walk_depmpo_h2190_gaussvel" "bl_walk_mpo_h2190_gaussvel"
    "bl_run_depmpo_h2190_gaussvel"  "bl_run_mpo_h2190_gaussvel"
)
YAMLS=(
    "${BASE}/walk_depmpo_onlyVelRew.yaml"  "${BASE}/walk_mpo_onlyVelRew.yaml"
    "${BASE}/run_depmpo_onlyVelRew.yaml"   "${BASE}/run_mpo_onlyVelRew.yaml"
    "${BASE}/walk_depmpo_gaussianVel.yaml" "${BASE}/walk_mpo_gaussianVel.yaml"
    "${BASE}/run_depmpo_gaussianVel.yaml"  "${BASE}/run_mpo_gaussianVel.yaml"
)

for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"
    yaml="${YAMLS[$i]}"
    if screen -list | grep -q "\.${name}[[:space:]]"; then
        echo "SKIP: screen '$name' already running"
        continue
    fi
    echo "START: $name"
    screen -dmS "$name" bash -c "
        source ~/anaconda3/etc/profile.d/conda.sh
        conda activate ${CONDA_ENV}
        cd ${PROJ_DIR}
        python -m deprl.main ${yaml}
        exec bash
    "
done
