#!/usr/bin/env bash
# ──────────────────────────────────────────────
#  CARES-Lite 启动脚本
#  在项目根目录运行：bash run.sh
# ──────────────────────────────────────────────
set -e

mkdir -p output

python -m src.main \
    --seed 42 \
    --num-clients 100 \
    --total-rounds 30 \
    --warmup-rounds 5 \
    --cluster-interval 5 \
    --min-cluster-size 5 \
    --probe-pool-size 16 \
    --probe-sigma 0.05 \
    --clip-norm 5.0 \
    --dpmm-prior 0.03 \
    --dpmm-max-components 10 \
    --client-frac 0.2 \
    --local-epochs 1 \
    --lr 0.02 \
    --output-dir ./output \
    --output-name results.csv

echo ""
echo "Done. Results saved in output/"
