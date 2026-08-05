#!/usr/bin/env bash
# 결합 실험 320 을 batch 16 으로 다시. 비교 대상(P2@320, s@320)이 batch 16 이라
# 8 로 돌린 것은 그대로 비교할 수 없다 - 배치는 고정 조건이다(1차 D8).
# 640 은 P2 격자가 커서 12GB 에 batch 16 이 안 들어간다 - 그건 단서로 남긴다.
set -u
cd /home/user/sod-project
while pgrep -f '0[7]_trai[n]' >/dev/null || pgrep -f 'run_comb[o]\.sh' >/dev/null; do sleep 60; done
echo "########## s+P2+열화 @ 320 (batch 16, 재실행) ##########"
~/venvs/sod/bin/python scripts/07_train.py --cfg configs/yolo11s-seg-p2.yaml \
    --model yolo11s-seg.pt --imgsz 320 --mask-ratio 4 --degrade --epochs 100 --batch 16 \
    --name yolo11s-p2_320_deg_b16 2>&1 | grep -avE "^\s*$"
echo "########## 재실행 완료 ##########"
