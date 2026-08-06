#!/usr/bin/env python3
"""10 - "줄여서 작아진 물체"와 "원래 작게 찍힌 물체"의 화질을 **크기를 맞춰** 비교한다.

09 에서 배운 것 — 사진 전체 통계로는 답이 안 나온다.
  - 고주파 비율은 크기를 줄이면 함께 올라간다(대조군 0.0118 -> 0.0236 -> 0.0465).
    같은 사진의 다른 크기끼리도 비교가 안 되는데 다른 출처끼리는 더 안 된다.
  - 평탄부 노이즈는 포화된 흰 배경이 표준편차 0 으로 잡혀 하위 백분위가 0 이 된다.

그래서 **물체 크기를 맞춘다.** E4-2 실측:
    대조군      imgsz 640 -> 11.6px  (리샘플링 없음. 자연히 작게 찍힌 것)
    screw-nut-bolt  imgsz 320 -> 12.5px  (줄여서 작아진 것)
    arg-fixings3    imgsz 320 -> 11.1px  (줄여서 작아진 것)
셋 다 12px 안팎이다. 이 상태에서 물체 주변을 잘라 재면 "줄인 것이 더 깨끗한가"에 답이 된다.

재는 것(물체마다)
  - 배경 노이즈 : 물체 박스 바깥 고리에서, 3x3 중앙값을 뺀 잔차의 표준편차.
                  포화 화소(<=2 또는 >=253)는 뺀다.
  - 경계 급함   : 박스 안 평균 |gradient| / 박스 안 표준편차.
                  대비로 나눠 밝기 차이를 상쇄한다. 흐릴수록 작아진다.
  - 대비        : 박스 안 표준편차
  - 도드라짐    : |박스 안 평균 - 고리 평균| / 고리 표준편차.
                  탐지가 실제로 의존하는 양이다 - 물체가 배경에서 얼마나 떨어져 보이는가.
  - 미헬슨 대비 : (max-min)/(max+min) 을 박스 안에서. 조명 세기에 덜 민감하다.

ASCII 로그.
"""
import argparse
import collections
import json
import math
import os
import random

import cv2
import numpy as np

# (역할, 출처, 이 크기로 레터박스) -> 물체가 12px 안팎이 되는 조합
TARGETS = [
    ("control", "rf-arg-fixings3", 640, "자연 (리샘플링 없음)"),
    ("train", "rf-screw-nut-bolt", 320, "축소"),
    ("train", "rf-arg-fixings3", 320, "축소"),
    # 비교용 - 같은 자료를 640 에서 (물체가 2배 큰 상태)
    ("train", "rf-screw-nut-bolt", 640, "참고: 축소 없음"),
]


def log(m):
    print(m, flush=True)


def stats(g, box, pad=1.6):
    """g: float32 회색조. box: (x1,y1,x2,y2) 정수."""
    h, w = g.shape
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < 3 or bh < 3:
        return None
    mx, my = int(bw * (pad - 1) / 2) + 2, int(bh * (pad - 1) / 2) + 2
    cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
    cx2, cy2 = min(w, x2 + mx), min(h, y2 + my)
    crop = g[cy1:cy2, cx1:cx2]
    if crop.size < 64:
        return None
    inner = np.zeros(crop.shape, bool)
    inner[y1 - cy1:y2 - cy1, x1 - cx1:x2 - cx1] = True
    ring = ~inner
    if ring.sum() < 32 or inner.sum() < 9:
        return None

    # 배경 노이즈 - 중앙값 필터 잔차. 포화 화소 제외.
    med = cv2.medianBlur(crop.astype(np.uint8), 3).astype(np.float32)
    res = crop - med
    ok = ring & (crop > 2) & (crop < 253)
    if ok.sum() < 32:
        return None
    noise = float(res[ok].std())

    # 경계 급함 - 대비로 정규화한 평균 기울기
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    sd = float(crop[inner].std())
    sharp = float(mag[inner].mean() / sd) if sd > 1e-6 else float("nan")

    # 도드라짐 - 배경 대비 물체의 분리도
    rin = crop[inner]; rout = crop[ok]
    rsd = float(rout.std())
    pop = float(abs(rin.mean() - rout.mean()) / rsd) if rsd > 1e-6 else float("nan")
    lo, hi = float(np.percentile(rin, 5)), float(np.percentile(rin, 95))
    mich = float((hi - lo) / (hi + lo)) if (hi + lo) > 1e-6 else float("nan")
    return {"noise": noise, "sharp": sharp, "contrast": sd, "pop": pop, "mich": mich}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=1500)
    ap.add_argument("--out", default="outputs/d_quality_gap.json")
    args = ap.parse_args()
    rng = random.Random(0)

    ann = {}
    for role in ["train", "control"]:
        p = "data/build/%s/annotations.coco.json" % role
        d = json.load(open(p, encoding="utf-8"))
        im = {i["id"]: i for i in d["images"]}
        by = collections.defaultdict(list)
        for a in d["annotations"]:
            by[a["image_id"]].append(a)
        ann[role] = (im, by)

    out = {}
    for role, src, sz, label in TARGETS:
        im, by = ann[role]
        ids = [i for i, info in im.items() if info.get("source") == src and by.get(i)]
        rng.shuffle(ids)
        rows, sizes = [], []
        for iid in ids:
            info = im[iid]
            fp = "data/build/%s/images/%s" % (role, info["file_name"])
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            oh, ow = img.shape[:2]
            r = sz / max(oh, ow)
            nh, nw = int(round(oh * r)), int(round(ow * r))
            g = cv2.resize(img, (nw, nh),
                           interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
                           ).astype(np.float32)
            for a in by[iid]:
                x, y, bw, bh = a["bbox"]
                box = (int(x * r), int(y * r), int(math.ceil((x + bw) * r)),
                       int(math.ceil((y + bh) * r)))
                s = stats(g, box)
                if s:
                    s["size"] = math.sqrt(bw * bh) * r
                    s["cls"] = ["bolt", "nut", "screw", "washer"][a["category_id"]]
                    rows.append(s)
                    sizes.append(s["size"])
            if len(rows) >= args.per_group:
                break
        key = "%s@%d" % (src.replace("rf-", ""), sz) + ("" if role != "control" else " (대조군)")
        out[key] = {"label": label, "rows": rows}
        log("%-34s n=%5d  물체 %.1fpx" % (key + " " + label, len(rows),
                                          sorted(sizes)[len(sizes) // 2] if sizes else 0))

    os.makedirs("outputs", exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)

    log("\n=== 물체 크기를 맞춘 화질 비교 (중앙값) ===")
    log("%-34s %6s %8s %8s %8s %8s %8s %8s"
        % ("대상", "n", "물체px", "배경노이즈", "경계급함", "대비", "도드라짐", "미헬슨"))
    for k, v in out.items():
        rows = v["rows"]
        if not rows:
            continue
        m = lambda f: sorted(x[f] for x in rows)[len(rows) // 2]
        log("%-34s %6d %8.1f %8.3f %8.3f %8.1f %8.3f %8.3f"
            % (k + " " + v["label"], len(rows), m("size"), m("noise"), m("sharp"),
               m("contrast"), m("pop"), m("mich")))
    log("\n=== 클래스별 도드라짐 (클래스 구성 탓인지 확인) ===")
    log("%-34s %s" % ("대상", "  ".join("%-10s" % c for c in ["bolt", "nut", "screw", "washer"])))
    for k, v in out.items():
        by = collections.defaultdict(list)
        for r in v["rows"]:
            by[r.get("cls", "?")].append(r["pop"])
        cells = []
        for c in ["bolt", "nut", "screw", "washer"]:
            z = sorted(by.get(c, []))
            cells.append("%-10s" % ("%.3f(%d)" % (z[len(z) // 2], len(z)) if z else "-"))
        log("%-34s %s" % (k, "  ".join(cells)))


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    main()
