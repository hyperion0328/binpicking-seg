#!/usr/bin/env bash
# 화질 축 3회 @ 640. 기준선은 yolo11s-seg_640 (이미 있음).
#  deg  = 기획서 원안 (노이즈·흐림·압축)
#  flat = E8 실측이 가리킨 것 (대비 축소·옅은 안개)
#  both = 둘 다
set -u
cd /home/user/sod-project
PY=~/venvs/sod/bin/python
run () {  # $1 이름, 나머지 인자
  echo "########## $1 ##########"
  shift
  $PY scripts/07_train.py --model yolo11s-seg.pt --imgsz 640 --epochs 100 --batch 16 "$@" 2>&1 | grep -avE "^\s*$"
}
run "deg @640"  --degrade          --name yolo11s-seg_640_deg
run "flat @640" --flatten          --name yolo11s-seg_640_flat
run "both @640" --degrade --flatten --name yolo11s-seg_640_both
echo "########## 화질 축 3회 완료 ##########"
