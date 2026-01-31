#!/bin/bash
#SBATCH --job-name=ooftest
#SBATCH --output=foo.out
#SBATCH -t 48:00:00  # Job time limit
#SBATCH --nodes=1  # Node count required for the job
#SBATCH --ntasks=1  # Number of tasks to be launched
#SBATCH --cpus-per-task=12  # Number of cores per task
#SBATCH --mem=14G  # Mem required per node
#SBATCH --parsable

set -euo pipefail

# 10 jinc parameter sets (lambda/D units) for grid search
r_max_vals=(1.5 1.5 1.5 1.5 1.8 2.0 2.0 1.5 1.5 2.0)
a1100_vals=("1.1,0.35,2.0" "1.1,0.40,2.0" "1.1,0.35,1.6" "1.1,0.35,2.4" "1.1,0.35,2.0" "1.1,0.32,2.0" "1.1,0.45,2.0" "1.05,0.35,2.0" "1.15,0.35,2.0" "1.1,0.38,2.2")
a1400_vals=("1.1,0.36,2.0" "1.1,0.41,2.0" "1.1,0.36,1.6" "1.1,0.36,2.4" "1.1,0.36,2.0" "1.1,0.33,2.0" "1.1,0.46,2.0" "1.05,0.36,2.0" "1.15,0.36,2.0" "1.1,0.39,2.2")
a2000_vals=("1.1,0.38,2.0" "1.1,0.43,2.0" "1.1,0.38,1.6" "1.1,0.38,2.4" "1.1,0.38,2.0" "1.1,0.35,2.0" "1.1,0.48,2.0" "1.05,0.38,2.0" "1.15,0.38,2.0" "1.1,0.41,2.2")

for idx in "${!r_max_vals[@]}"; do
    redu=$(printf "redu%02d" "$((idx + 10))")

    srun tolteca reduce \
        --steps.0.enabled true \
        --steps.0.config.low_level.mapmaking.method jinc \
        --steps.0.config.low_level.mapmaking.jinc_filter.r_max "${r_max_vals[$idx]}" \
        --steps.0.config.low_level.mapmaking.jinc_filter.shape_params.a1100 "[${a1100_vals[$idx]}]" \
        --steps.0.config.low_level.mapmaking.jinc_filter.shape_params.a1400 "[${a1400_vals[$idx]}]" \
        --steps.0.config.low_level.mapmaking.jinc_filter.shape_params.a2000 "[${a2000_vals[$idx]}]"
done
