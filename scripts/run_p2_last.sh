#!/usr/bin/env bash
# 마지막 칸: P2 mr2 @ 160. 앞 학습이 끝나면 시작한다.
# 주의: 대기 조건에 자기 명령줄과 일치하는 문자열을 쓰면 자신을 보고 영원히 기다린다.
# 그래서 문자 클래스로 자기 일치를 피한다.
set -u
cd /home/user/sod-project
while pgrep -f '0[7]_train' >/dev/null; do sleep 30; done
echo "########## (재실행) P2 mr2 @ 160 ##########"
~/venvs/sod/bin/python scripts/07_train.py --cfg configs/yolo11s-seg-p2.yaml \
    --model yolo11s-seg.pt --imgsz 160 --mask-ratio 2 --epochs 100 --batch 16 \
    --name yolo11s-p2_160_mr2 2>&1 | grep -avE "^\s*$"
echo "########## 2x2 설계 완료 ##########"
