#!/usr/bin/env python3
"""02 - SAM 을 도구로 쓰기 전에 측정 대상으로 세운다.

정답 마스크가 이미 있는 인스턴스에 대해 정답 박스를 프롬프트로 주고 SAM 마스크를
받아, 정답 마스크와의 IoU 를 객체 크기별·클래스별로 잰다.

  - washer 는 전량(117개) 넣는다. 채워야 할 대상이 바로 이 클래스다.
  - 나머지 클래스는 크기 구간별로 층화 표집한다. washer 마스크가 큰 쪽에 몰려 있어
    (중앙값 63.6px) 작은 구간의 성능은 다른 클래스로만 알 수 있다.

결과가 나쁘면 채우지 않는다 — 그 실측이 곧 근거다.
ASCII 로그.
"""
import argparse
import collections
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np
from pycocotools import mask as maskutil

CL = ["bolt", "nut", "screw", "washer"]
ROLES = ["train", "val", "test", "control"]
BUCKETS = [(0, 16), (16, 24), (24, 32), (32, 48), (48, 72), (72, 1e9)]
PER_BUCKET = 90            # washer 외 클래스에서 (클래스 x 구간) 당 최대 표본


def log(m):
    print(m, flush=True)


def bucket(s):
    for i, (a, b) in enumerate(BUCKETS):
        if a <= s < b:
            return i
    return len(BUCKETS) - 1


def bname(i):
    a, b = BUCKETS[i]
    return "%d-%d" % (a, b) if b < 1e9 else "%d+" % a


def gt_mask(a, h, w):
    """COCO segmentation -> (h,w) uint8"""
    s = a.get("segmentation")
    if isinstance(s, dict):
        ss = dict(s)
        if isinstance(ss.get("counts"), str):
            ss["counts"] = ss["counts"].encode()
        return maskutil.decode(ss)
    if isinstance(s, list) and s:
        rles = maskutil.frPyObjects(s, h, w)
        return maskutil.decode(maskutil.merge(rles))
    return None


def collect(seed=0):
    """(이미지 경로, [인스턴스…]) 목록을 만든다."""
    rng = random.Random(seed)
    pool = collections.defaultdict(list)      # (cls, bucket) -> [item]
    for role in ROLES:
        p = "data/build/%s/annotations.coco.json" % role
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        im = {i["id"]: i for i in d["images"]}
        for a in d["annotations"]:
            if not a.get("has_mask"):
                continue
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            info = im[a["image_id"]]
            pool[(a["category_id"], bucket(math.sqrt(w * h)))].append({
                "path": "data/build/%s/images/%s" % (role, info["file_name"]),
                "iw": info["width"], "ih": info["height"],
                "bbox": [x, y, x + w, y + h], "size": math.sqrt(w * h),
                "cls": a["category_id"], "seg": a["segmentation"],
                "source": info.get("source", "?"), "role": role,
            })
    picked = []
    for (c, b), items in sorted(pool.items()):
        take = items if CL[c] == "washer" else rng.sample(items, min(PER_BUCKET, len(items)))
        picked += take
    byimg = collections.defaultdict(list)
    for it in picked:
        byimg[it["path"]].append(it)
    return byimg, picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sam_b.pt")
    ap.add_argument("--limit-images", type=int, default=0)
    args = ap.parse_args()

    from ultralytics import SAM

    byimg, picked = collect()
    log("표본 %d개 / 이미지 %d장 (모델 %s)" % (len(picked), len(byimg), args.model))
    dist = collections.Counter((CL[i["cls"]], bname(bucket(i["size"]))) for i in picked)
    for c in CL:
        row = "  %-7s " % c + " ".join("%s:%d" % (bname(b), dist.get((c, bname(b)), 0))
                                       for b in range(len(BUCKETS)))
        log(row)

    model = SAM(args.model)
    paths = sorted(byimg)
    if args.limit_images:
        paths = paths[:args.limit_images]

    rows = []
    t0 = time.time()
    for n, p in enumerate(paths, 1):
        items = byimg[p]
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = [it["bbox"] for it in items]
        try:
            res = model(p, bboxes=boxes, verbose=False)
        except Exception as e:
            log("  실패 %s : %s" % (os.path.basename(p), str(e)[:90]))
            continue
        md = res[0].masks
        if md is None:
            continue
        pred = md.data.cpu().numpy().astype(np.uint8)
        for k, it in enumerate(items):
            if k >= len(pred):
                break
            g = gt_mask({"segmentation": it["seg"]}, h, w)
            if g is None or g.sum() == 0:
                continue
            pm = pred[k]
            if pm.shape != g.shape:
                pm = cv2.resize(pm, (g.shape[1], g.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            inter = np.logical_and(pm, g).sum()
            union = np.logical_or(pm, g).sum()
            rows.append({"cls": CL[it["cls"]], "size": it["size"],
                         "bucket": bname(bucket(it["size"])),
                         "iou": float(inter / union) if union else 0.0,
                         "source": it["source"], "role": it["role"]})
        if n % 40 == 0:
            log("  %d/%d 이미지 · %d개 측정 · %.0fs" % (n, len(paths), len(rows), time.time() - t0))

    os.makedirs("outputs", exist_ok=True)
    tag = args.model.replace(".pt", "")
    with open("outputs/d_sam_iou_%s.jsonl" % tag, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log("\n=== 결과: SAM(%s) 마스크 vs 정답 마스크 IoU ===" % tag)
    log("%-8s %-8s %6s %7s %7s %7s %7s" % ("클래스", "크기", "n", "중앙값", "평균", ">0.7", ">0.9"))
    def agg(sel, c, b):
        v = sorted(x["iou"] for x in sel)
        if not v:
            return
        log("%-8s %-8s %6d %7.3f %7.3f %6.0f%% %6.0f%%"
            % (c, b, len(v), v[len(v) // 2], sum(v) / len(v),
               100 * sum(1 for x in v if x > .7) / len(v),
               100 * sum(1 for x in v if x > .9) / len(v)))
    for c in CL:
        for b in range(len(BUCKETS)):
            agg([r for r in rows if r["cls"] == c and r["bucket"] == bname(b)], c, bname(b))
    log("-" * 56)
    for b in range(len(BUCKETS)):
        agg([r for r in rows if r["bucket"] == bname(b)], "전체", bname(b))
    agg(rows, "전체", "전구간")
    log("\n원자료: outputs/d_sam_iou_%s.jsonl (%d행, %.0f초)" % (tag, len(rows), time.time() - t0))


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    main()
