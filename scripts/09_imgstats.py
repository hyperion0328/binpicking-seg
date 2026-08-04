#!/usr/bin/env python3
"""09 - "줄인 사진은 실제보다 깨끗하다" 를 재본다. 증강을 설계하기 전에.

기획서 04절이 이 축의 근거로 삼은 문장인데 아직 실측한 적이 없다. 재보지 않고 노이즈·흐림·
압축을 켜면, 효과가 없을 때 "증강이 안 듣는다"인지 "애초에 그런 격차가 없었다"인지 모른다.
1차 D9 에서 CAM 집중도를 검증 없이 쓴 것과 같은 실수가 된다.

모델이 보는 상태(레터박스 640)에서 잰다. 대조군만 배율 1.00 이고 나머지는 확대/축소를 겪는다.

  - 노이즈    : **평탄부에서만** 잰다. 전역 라플라시안 추정은 나사산 같은 질감을 노이즈로
                세어, 세밀한 사진을 축소한 것이 "거칠다"고 잘못 나온다(첫 측정에서 실제로 그랬다).
                8x8 블록 국소 표준편차의 하위 5% 를 쓴다.
  - 선명도    : 라플라시안 분산, 그리고 FFT 고주파 비율(전체 대비 상위 1/4 대역 에너지).
  - 블록 자국 : JPEG 8x8 경계에서의 계단 정도. 경계 화소차 / 비경계 화소차.

패딩 영역은 뺀다. ASCII 로그.
"""
import argparse
import collections
import glob
import json
import math
import os
import random

import cv2
import numpy as np

# Immerkaer 커널
K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def log(m):
    print(m, flush=True)


def letterbox(img, size=640):
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (nw, nh), interpolation=interp), r


def noise_sigma(g, blk=8, pct=5):
    """평탄부 노이즈. 8x8 블록의 국소 표준편차 중 하위 5% 를 쓴다.

    전역 추정(Immerkaer 등)은 질감을 노이즈로 센다. 부품 사진은 나사산·널링이 많아
    "세밀한 원본을 축소한 것"이 오히려 노이즈가 큰 것으로 나온다. 평탄부로 제한해야
    리샘플링이 실제로 노이즈를 지웠는지 볼 수 있다.
    """
    h, w = g.shape
    h -= h % blk; w -= w % blk
    if h < blk or w < blk:
        return float("nan")
    b = g[:h, :w].reshape(h // blk, blk, w // blk, blk).transpose(0, 2, 1, 3)
    sd = b.reshape(-1, blk * blk).std(axis=1)
    return float(np.percentile(sd, pct))


def hf_ratio(g):
    """FFT 에서 상위 1/4 대역이 차지하는 에너지 비율. 낮을수록 뭉개진 사진."""
    f = np.fft.fftshift(np.fft.fft2(g - g.mean()))
    p = np.abs(f) ** 2
    h, w = g.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    rad = np.sqrt(((y - cy) / max(1, cy)) ** 2 + ((x - cx) / max(1, cx)) ** 2)
    tot = p.sum()
    return float(p[rad > 0.5].sum() / tot) if tot > 0 else float("nan")


def blockiness(g):
    """JPEG 8x8 경계의 계단. 1.0 이면 자국 없음."""
    d = np.abs(np.diff(g, axis=1))
    if d.shape[1] < 16:
        return float("nan")
    idx = np.arange(d.shape[1])
    on = d[:, (idx % 8) == 7]
    off = d[:, (idx % 8) != 7]
    return float(on.mean() / off.mean()) if off.mean() > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--per-group", type=int, default=120)
    ap.add_argument("--out", default="outputs/d_imgstats.json")
    ap.add_argument("--sizes", default="640,320,160",
                    help="같은 사진을 이 크기들로 각각 줄여 재본다 - 내용은 그대로 두고 배율만 바꾼다")
    args = ap.parse_args()
    rng = random.Random(0)

    # 역할/출처별로 사진을 모은다. 대조군은 따로 센다.
    groups = collections.defaultdict(list)
    for role in ["train", "val", "test", "control", "crossdomain"]:
        p = "data/build/%s/annotations.coco.json" % role
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        for i in d["images"]:
            fp = "data/build/%s/images/%s" % (role, i["file_name"])
            key = "control" if role == "control" else i.get("source", "?")
            groups[key].append(fp)

    rows = collections.defaultdict(list)
    for key, files in sorted(groups.items()):
        pick = rng.sample(files, min(args.per_group, len(files)))
        for fp in pick:
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            oh, ow = img.shape[:2]
            rec = {"src_h": oh, "src_w": ow}
            for sz in [int(x) for x in args.sizes.split(",")]:
                lb, r = letterbox(img, sz)
                g = lb.astype(np.float32)
                rec["%d" % sz] = {"ratio": r, "sigma": noise_sigma(g), "hf": hf_ratio(g),
                                  "lapvar": float(cv2.Laplacian(g, cv2.CV_32F).var()),
                                  "block": blockiness(g)}
            rows[key].append(rec)

    os.makedirs("outputs", exist_ok=True)
    json.dump({k: v for k, v in rows.items()}, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False)

    sizes = [int(x) for x in args.sizes.split(",")]
    med = lambda v, sz, k: sorted(x["%d" % sz][k] for x in v)[len(v) // 2]
    order = sorted(rows, key=lambda k: (k != "control", k))
    for k in ["sigma", "hf", "block"]:
        nm = {"sigma": "평탄부 노이즈", "hf": "고주파 비율", "block": "블록 자국"}[k]
        log("\n=== %s (중앙값) - 같은 사진을 크기만 바꿔 잰다 ===" % nm)
        log("%-20s %5s " % ("출처", "n") + " ".join("%12s" % ("%d(x%s)" % (sz, "?")) for sz in sizes))
        for g in order:
            v = rows[g]
            if not v:
                continue
            rs = " ".join("%12s" % ("%.4f (x%.2f)" % (med(v, sz, k), med(v, sz, "ratio")))
                          for sz in sizes)
            log("%-20s %5d %s" % (g, len(v), rs))
    log("\n원자료: %s" % args.out)

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    main()
