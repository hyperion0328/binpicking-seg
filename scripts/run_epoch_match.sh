#!/usr/bin/env bash
# P2 의 박스 이득을 epoch 맞춰 재확인한다(E14-3).
#
# 지금 근거는 160·320 에서 P2 쪽이 각각 +17 epoch 더 돌았고, 교란은 P2 에 유리한 방향이다.
# epoch 가 맞는 유일한 비교(640)에서는 P2 가 졌다. 세 자리를 전부 100 epoch 로 맞춰
# "P2 가 소형에서 이긴다"가 확정되는지 본다.
#
#   yolo11s-seg_160_e100      기본 헤드 160  (원본 83ep)
#   yolo11s-seg_320_e100      기본 헤드 320  (원본 78ep)
#   yolo11s-p2_320_mr4_e100   P2 320         (원본 95ep)
#   (P2 160 은 원본이 이미 100ep 이라 그대로 쓴다)
#
# 인자는 원본(run_curve.sh · run_p2.sh)과 같고 `--patience 0` 만 더한다.
# 원본 런은 증거라 이름을 새로 준다 - 07_train.py 는 같은 이름이면 디렉터리를 지운다.
#
# **이 스크립트는 몇 번을 다시 돌려도 안전하다.** 100 epoch 을 완주하고 best.pt 가 있는
# 런은 건너뛴다. VS Code 를 껐다 켜서 중간에 끊겼으면 그냥 다시 실행하면 이어진다.
set -u
cd /home/user/sod-project
PY=~/venvs/sod/bin/python
STATE=.omc/state/epoch-match.status
mkdir -p .omc/state .omc/logs

log() { echo "[$(date '+%H:%M:%S')] $*"; }
say() { echo "$1" > "$STATE"; }

# 다른 학습이 돌고 있으면 기다린다(GPU 를 겹쳐 쓰면 메모리 부족으로 죽는다).
# 셸 이름이 아니라 파이썬 학습 프로세스만 본다 - 자기 자신에 걸리면 영원히 안 끝난다.
while pgrep -f '0[7]_trai[n]' >/dev/null; do
  say "대기: 다른 학습이 끝나기를 기다리는 중"
  sleep 60
done

done_already () {   # $1 = 런 이름. 100 epoch 완주 + best.pt 있으면 0
  local n=$1 csv="runs/seg/$1/results.csv"
  [ -f "runs/seg/$n/weights/best.pt" ] || return 1
  [ -f "$csv" ] || return 1
  [ "$(($(wc -l < "$csv") - 1))" -ge 100 ] || return 1
}

train () {          # $1 = 이름, 나머지 = 07_train.py 인자
  local name=$1; shift
  if done_already "$name"; then
    log "건너뜀 (이미 100 epoch 완주): $name"
    return 0
  fi
  log "학습 시작: $name"
  say "학습 중: $name"
  $PY scripts/07_train.py "$@" --epochs 100 --batch 16 --patience 0 \
      --name "$name" 2>&1 | grep -avE "^\s*$"
  if done_already "$name"; then
    log "완료: $name"
  else
    log "*** 실패 또는 중단: $name — 스크립트를 다시 실행하면 이 런부터 재개한다"
    say "실패: $name"
    return 1
  fi
}

RUNS="yolo11s-seg_160_e100 yolo11s-seg_320_e100 yolo11s-p2_320_mr4_e100"

train yolo11s-seg_160_e100 \
  --model yolo11s-seg.pt --imgsz 160 || exit 1
train yolo11s-seg_320_e100 \
  --model yolo11s-seg.pt --imgsz 320 || exit 1
train yolo11s-p2_320_mr4_e100 \
  --cfg configs/yolo11s-seg-p2.yaml --model yolo11s-seg.pt --imgsz 320 --mask-ratio 4 || exit 1

log "세 런 모두 완주 — 평가 시작"
say "평가 중"
ONLY=$(echo $RUNS | tr ' ' ',')
$PY scripts/08_eval.py --only "$ONLY" --out outputs/d_eval_native.json 2>&1 | grep -avE "^\s*$"

say "완료"
log "########## epoch 맞춤 재확인 완료 ##########"
