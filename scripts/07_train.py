#!/usr/bin/env python3
"""07 - 학습. 고정 조건은 기획서 07절, 축은 인자로 준다.

고정 조건(모든 런에서 같아야 비교가 성립한다)
  seed 0 / deterministic / **배치 고정**(AutoBatch 금지 - 1차에서 같은 데이터로 mAP 0.345 vs
  0.282 가 나온 원인) / 학습 imgsz = 평가 imgsz / 레터박스 / COCO 사전학습에서 출발.

증강(기획서 D12·D13)
  회전·상하뒤집기 **켬**(톱다운이라 방향이 자유롭다) / mixup·cutmix **끔**(이미 겹친 장면에
  인위적 겹침을 더하면 원인 구분이 안 된다) / 색·명암은 기본값 유지(금속 반사가 재질 단서).
  copy-paste 는 **원 논문의 짝과 함께** 켠다 —
    - Ghiasi(CVPR 2021): 다른 사진에서 오려 붙이고(`copy_paste_mode=mixup`),
      **크게 흔드는 크기 변형(LSJ 0.1~2.0)** 과 짝지을 때 효과가 크다.
      ultralytics 기본 `scale=0.5`(±50%)는 폭이 좁아 튜플로 넓힌다.
      곡선의 모든 점에서 같은 값을 쓰므로 점끼리의 비교는 훼손되지 않는다.
    - Kisantal(2019): copy-paste 와 **소형 객체가 담긴 사진의 샘플링 비중 올리기**가 한 쌍이다.
      도구에 없는 기능이라 목록 파일에 경로를 반복해 넣어 구현한다(`--oversample`).

발산 감지 — fp16 오버플로로 조용히 NaN 이 되면 ultralytics 는 계속 돌며 발산 직전 best.pt 를
남겨 "그냥 성능 낮은 런"과 구별이 안 된다. results.csv 를 보고 자동으로 `amp=False` 재학습한다.

ASCII 로그.
"""
import argparse
import collections
import json
import math
import os
import shutil
import sys

SMALL_PX = 32          # COCO 소형 기준. 640 환산 크기로 잰다
OVERSAMPLE = [(0.80, 3), (0.50, 2)]   # (소형 비율 하한, 반복 횟수). 나머지는 1회


def log(m):
    print(m, flush=True)


def build_oversampled_list(out_txt, imgsz):
    """소형 객체가 많은 사진을 목록에 여러 번 넣는다(Kisantal 후반부)."""
    d = json.load(open("data/build/train/annotations.filled.coco.json", encoding="utf-8"))
    im = {i["id"]: i for i in d["images"]}
    per = collections.defaultdict(list)
    for a in d["annotations"]:
        info = im[a["image_id"]]
        r = imgsz / max(info["width"], info["height"])
        x, y, w, h = a["bbox"]
        per[a["image_id"]].append(math.sqrt(w * h) * r)

    base = [l for l in open("data/yolo/train_box.txt").read().split("\n") if l.strip()]
    byname = {os.path.basename(p): p for p in base}
    lines, rep = [], collections.Counter()
    src_cnt = collections.Counter()
    for iid, info in im.items():
        p = byname.get(info["file_name"])
        if p is None:
            continue
        sz = per.get(iid, [])
        frac = sum(1 for s in sz if s < SMALL_PX) / len(sz) if sz else 0.0
        n = 1
        for lo, k in OVERSAMPLE:
            if frac >= lo:
                n = k
                break
        lines += [p] * n
        rep[n] += 1
        src_cnt[(info.get("source", "?"), n)] += 1
    open(out_txt, "w").write("\n".join(lines) + "\n")
    log("소형 비중 올리기 - 사진 %d장 -> 목록 %d줄" % (sum(rep.values()), len(lines)))
    for n in sorted(rep, reverse=True):
        log("  %d회 반복 %4d장" % (n, rep[n]))
    log("  출처별:")
    for k in sorted(src_cnt):
        log("    %-20s %d회 %4d장" % (k[0], k[1], src_cnt[k]))
    return out_txt


def write_yaml(path, train_list, val_list):
    root = os.path.abspath("data/yolo")
    open(path, "w").write(
        "# scripts/07_train.py 생성\npath: %s\ntrain: %s\nval: %s\nnames:\n"
        "  0: bolt\n  1: nut\n  2: screw\n  3: washer\n" % (root, train_list, val_list))
    return path


def diverged(csv_path):
    """results.csv 에 NaN 이 있으면 발산으로 본다."""
    if not os.path.exists(csv_path):
        return False
    t = open(csv_path).read().lower()
    return "nan" in t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n-seg.pt")
    ap.add_argument("--cfg", default=None,
                    help="모델 yaml(P2 등). 주면 이 구조로 만들고 --model 가중치를 옮겨 싣는다")
    ap.add_argument("--mask-ratio", type=int, default=4,
                    help="정답 마스크 해상도 = imgsz/이 값. P2 는 proto 가 imgsz/2 라 2 로 맞춰야 "
                         "한다 - 안 맞추면 ultralytics 가 proto 를 정답 쪽으로 낮춰 P2 의 이득이 사라진다")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)      # 고정. -1 금지
    ap.add_argument("--name", default=None)
    ap.add_argument("--oversample", type=int, default=1)  # 1=켬
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    if args.batch <= 0:
        sys.exit("배치는 고정값이어야 한다. -1(AutoBatch)은 판정을 불가능하게 만든다.")

    base = os.path.splitext(os.path.basename(args.cfg or args.model))[0]
    tag = args.name or "%s_%d_mr%d" % (base, args.imgsz, args.mask_ratio)
    train_list = "train_box.txt"
    if args.oversample:
        # 목록은 **항상 640 기준으로** 만든다. imgsz 마다 다시 계산하면 축소 곡선에서
        # imgsz 와 학습 자료 구성이 동시에 바뀌어 무엇이 성능을 움직였는지 갈리지 않는다.
        train_list = "train_os640.txt"
        build_oversampled_list(os.path.join("data/yolo", train_list), 640)
    yml = write_yaml("data/yolo/run_%s.yaml" % tag, train_list, "val_seg.txt")

    from ultralytics import YOLO
    common = dict(
        data=yml, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
        seed=0, deterministic=True, device=args.device, workers=args.workers,
        mask_ratio=args.mask_ratio,
        project=os.path.abspath("runs/seg"), exist_ok=True, patience=30, val=True, plots=True,
        # --- 증강 (D12 / D13) ---
        degrees=180.0, flipud=0.5, fliplr=0.5,
        scale=(0.1, 2.0),                       # Ghiasi LSJ. 기본 0.5 는 폭이 좁다
        copy_paste=0.5, copy_paste_mode="mixup",   # 다른 사진에서 오려 붙이기 = 원 논문 방식
        mixup=0.0, cutmix=0.0,                  # 이미 겹친 장면에 인위적 겹침을 더하지 않는다
    )
    log("=== %s / imgsz %d / batch %d / epochs %d / mask_ratio %d ==="
        % (args.cfg or args.model, args.imgsz, args.batch, args.epochs, args.mask_ratio))
    log("데이터 %s (train=%s)" % (yml, train_list))

    for amp in (True, False):
        name = tag if amp else tag + "_noamp"
        d = os.path.join(os.path.abspath("runs/seg"), name)
        if os.path.isdir(d):
            shutil.rmtree(d)
        trainer = None
        if args.cfg:
            # 구조는 yaml, 가중치는 COCO 사전학습본에서 겹치는 층만 옮겨 싣는다.
            # 안 그러면 P2 만 맨바닥에서 시작해 "P2 가 나쁘다"가 아니라 "출발점이 다르다"를 재게 된다.
            y = YOLO(args.cfg).load(args.model)
            # ultralytics 검증기는 "proto 는 imgsz/4" 를 상수로 박아둬 P2 에서 죽는다.
            # 상수를 stride 에서 계산하도록 바꾼 검증기를 쓴다(sodseg/p2_seg.py).
            from sodseg.p2_seg import P2SegmentationTrainer
            trainer = P2SegmentationTrainer
        else:
            y = YOLO(args.model)
        y.train(name=name, amp=amp, **({"trainer": trainer} if trainer else {}), **common)
        # 저장 위치를 추측하지 않고 트레이너에게 묻는다.
        # 상대 경로 project 는 runs_dir 밑으로 다시 들어가 NaN 검사가 헛돈 적이 있다.
        d = str(getattr(y.trainer, "save_dir", d))
        csv = os.path.join(d, "results.csv")
        if not os.path.exists(csv):
            log("*** results.csv 를 못 찾았다: %s — 발산 검사를 못 한다 ***" % csv)
            return
        if not diverged(csv):
            log("\n완료: %s" % d)
            return
        log("\n*** NaN 발산 감지 (amp=%s) — amp=False 로 재학습한다 ***" % amp)
        if not amp:
            log("amp=False 에서도 발산했다. 학습률·배치를 손대야 한다.")
            return


if __name__ == "__main__":
    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    os.chdir(ROOT)
    sys.path.insert(0, os.path.abspath(ROOT))     # sodseg 를 찾게 한다
    main()
