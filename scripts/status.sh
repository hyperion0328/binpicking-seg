#!/usr/bin/env bash
# 지금 무엇이 돌고 있고 어디까지 갔는지 한눈에. VS Code 를 껐다 켠 뒤 이것부터 실행한다.
set -u
cd /home/user/sod-project

RUNS="yolo11s-seg_160_e100 yolo11s-seg_320_e100 yolo11s-p2_320_mr4_e100"

echo "=== 학습 프로세스 ==="
# 데이터로더 워커가 같은 명령줄로 여러 개 뜨므로 --name 기준으로 접어서 보여준다.
if pgrep -f '0[7]_trai[n]' >/dev/null; then
  pgrep -af '0[7]_trai[n]' \
    | sed -n 's/.*--name \([^ ]*\).*/\1/p' | sort | uniq -c \
    | while read -r n name; do printf "  %-28s (프로세스 %d개: 본체 1 + 워커)\n" "$name" "$n"; done
else
  echo "  (없음)"
fi

echo
echo "=== 진행 상태 ==="
if [ -f .omc/state/epoch-match.status ]; then echo "  $(cat .omc/state/epoch-match.status)"; else echo "  (기록 없음)"; fi

echo
echo "=== epoch 맞춤 재확인 3런 ==="
for n in $RUNS; do
  csv="runs/seg/$n/results.csv"
  if [ -f "$csv" ]; then
    e=$(( $(wc -l < "$csv") - 1 ))
    t=$(tail -1 "$csv" | cut -d, -f2)
    if [ -f "runs/seg/$n/weights/best.pt" ] && [ "$e" -ge 100 ]; then
      printf "  %-26s %3d/100  완료 (%.0f분)\n" "$n" "$e" "$(echo "$t/60" | bc -l 2>/dev/null || echo 0)"
    else
      printf "  %-26s %3d/100  진행 중\n" "$n" "$e"
    fi
  elif [ -d "runs/seg/$n" ]; then
    printf "  %-26s   -/100  시작됨 (첫 epoch 끝나면 숫자가 뜬다)\n" "$n"
  else
    printf "  %-26s   -/100  대기\n" "$n"
  fi
done

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi 없음)"

echo
echo "끊겼으면 이 한 줄로 이어서 돌린다(끝난 런은 건너뛴다):"
echo "  setsid nohup bash scripts/run_epoch_match.sh > .omc/logs/epoch_match.log 2>&1 < /dev/null &"
echo "로그:  tail -f .omc/logs/epoch_match.log"
