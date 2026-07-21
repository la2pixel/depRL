#!/bin/bash
# Usage: bash launch.sh [full|onlyvel|gaussvel]
# Omit argument to run all groups (not recommended — high RAM usage).
CONDA_ENV="lalitha"
PROJ_DIR="/home/nadinebadie/lalitha/depRL-sconegym"
BASE="experiments/hyfydy/baseline_runs/h1622"

GROUP="${1:-all}"

NAMES_FULL=(
    "bl_walk_depmpo_h1622" "bl_walk_mpo_h1622"
    "bl_run_depmpo_h1622"  "bl_run_mpo_h1622"
)
YAMLS_FULL=(
    "${BASE}/walk_depmpo_baseline.yaml" "${BASE}/walk_mpo_baseline.yaml"
    "${BASE}/run_depmpo_baseline.yaml"  "${BASE}/run_mpo_baseline.yaml"
)

NAMES_ONLYVEL=(
    "bl_walk_depmpo_h1622_velrew" "bl_walk_mpo_h1622_velrew"
    "bl_run_depmpo_h1622_velrew"  "bl_run_mpo_h1622_velrew"
)
YAMLS_ONLYVEL=(
    "${BASE}/walk_depmpo_onlyVelRew.yaml" "${BASE}/walk_mpo_onlyVelRew.yaml"
    "${BASE}/run_depmpo_onlyVelRew.yaml"  "${BASE}/run_mpo_onlyVelRew.yaml"
)

NAMES_GAUSSVEL=(
    "bl_walk_depmpo_h1622_gaussvel" "bl_walk_mpo_h1622_gaussvel"
    "bl_run_depmpo_h1622_gaussvel"  "bl_run_mpo_h1622_gaussvel"
)
YAMLS_GAUSSVEL=(
    "${BASE}/walk_depmpo_gaussianVel.yaml" "${BASE}/walk_mpo_gaussianVel.yaml"
    "${BASE}/run_depmpo_gaussianVel.yaml"  "${BASE}/run_mpo_gaussianVel.yaml"
)

case "$GROUP" in
    full)     NAMES=("${NAMES_FULL[@]}");     YAMLS=("${YAMLS_FULL[@]}") ;;
    onlyvel)  NAMES=("${NAMES_ONLYVEL[@]}");  YAMLS=("${YAMLS_ONLYVEL[@]}") ;;
    gaussvel) NAMES=("${NAMES_GAUSSVEL[@]}"); YAMLS=("${YAMLS_GAUSSVEL[@]}") ;;
    all)      NAMES=("${NAMES_FULL[@]}" "${NAMES_ONLYVEL[@]}" "${NAMES_GAUSSVEL[@]}")
              YAMLS=("${YAMLS_FULL[@]}" "${YAMLS_ONLYVEL[@]}" "${YAMLS_GAUSSVEL[@]}") ;;
    *)        echo "Usage: bash launch.sh [full|onlyvel|gaussvel]"; exit 1 ;;
esac

echo "Launching group: $GROUP (${#NAMES[@]} runs)"
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
