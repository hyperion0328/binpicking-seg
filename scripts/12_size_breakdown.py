#!/usr/bin/env python3
"""12 - 물체 크기 구간별로 얼마나 찾는가. "소형 부품을 잘 잡는가"에 답하는 자리.

mAP 합산으로는 답이 안 나온다. test 인스턴스의 12px 미만이 2% 뿐이고 42%가 32px 이상이라,
전체 mask mAP50 0.96 은 **주로 16px 이상 물체에서 얻은 점수**다. 대조군은 반대로 55%가
12px 미만이다. 그래서 **같은 크기 구간끼리** 비교해야 "소형에 강한가"를 알 수 있다.

재는 것(정답 물체마다)
  - 찾았는가 : 같은 클래스 예측과 박스 IoU >= 0.5 인 것이 있는가 (재현율)
  - 마스크 품질 : 찾은 것에 한해 예측 마스크와 정답 마스크의 IoU (정답 마스크가 있을 때만)

conf 는 전 모델 고정(기본 0.25). 재현율만 보므로 낮추면 올라가는 값이라 비교하려면 고정해야 한다.
ASCII 로그.
"""
import argparse
import collections
import json
import math
import os
import re

import cv2
import numpy as np
from pycocotools import mask as maskutil

CL = ["bolt", "nut", "screw", "washer"]
B = [(0, 8), (8, 12), (12, 16), (16, 24), (24, 32), (32, 1e9)]


def bn(i):
    a, b = B[i]
    return "%d-%d" % (a, b) if b < 1e9 else "%d+" % a


def bk(s):
    for i, (a, b) in enumerate(B):
        if a <= s < b:
            return i
    return len(B) - 1


def log(m):
    print(m, flush=True)


def gtm(seg, h, w):
    if isinstance(seg, dict):
        s = dict(seg)
        if isinstance(s.get("counts"), str):
            s["counts"] = s["counts"].encode()
        return maskutil.decode(s)
    if isinstance(seg, list) and seg:
        return maskutil.decode(maskutil.merge(maskutil.frPyObjects(seg, h, w)))
    return None


def iou_box(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/seg/yolo11s-seg_640_deg")
    ap.add_argument("--roles", default="test,control")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/d_size_breakdown.json")
    args = ap.parse_args()
    from ultralytics import YOLO

    res = json.load(open(args.out, encoding="utf-8")) if os.path.exists(args.out) else {}
    for run in args.runs.split(","):
        run = run.strip()
        name = os.path.basename(run)
        sz = int(re.search(r"_(\d+)", name).group(1))
        model = YOLO(os.path.join(run, "weights", "best.pt"))
        res.setdefault(name, {"imgsz": sz})
        for role in args.roles.split(","):
            role = role.strip()
            d = json.load(open("data/build/%s/annotations.coco.json" % role, encoding="utf-8"))
            im = {i["id"]: i for i in d["images"]}
            by = collections.defaultdict(list)
            for a in d["annotations"]:
                by[a["image_id"]].append(a)
            cnt = collections.Counter()
            hit = collections.Counter()
            mio = collections.defaultdict(list)
            ccnt, chit = collections.Counter(), collections.Counter()
            for n, (iid, anns) in enumerate(sorted(by.items()), 1):
                info = im[iid]
                fp = "data/build/%s/images/%s" % (role, info["file_name"])
                r = model(fp, imgsz=sz, conf=args.conf, device=args.device, verbose=False)[0]
                pb = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
                pc = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros(0, int)
                pm = r.masks.data.cpu().numpy().astype(np.uint8) if r.masks is not None else None
                H, W = info["height"], info["width"]
                # 640 환산 크기로 구간을 나눈다 - 소스마다 원본 해상도가 달라 원본 px 로는 못 비교한다
                sc = 640 / max(H, W)
                for a in anns:
                    x, y, bw, bh = a["bbox"]
                    g = [x, y, x + bw, y + bh]
                    s = math.sqrt(bw * bh) * sc
                    b = bn(bk(s))
                    cnt[b] += 1
                    c = CL[a["category_id"]]
                    ccnt[c] += 1
                    best, bi = 0.0, -1
                    for k in range(len(pb)):
                        if pc[k] != a["category_id"]:
                            continue
                        v = iou_box(g, pb[k])
                        if v > best:
                            best, bi = v, k
                    if best >= 0.5:
                        hit[b] += 1
                        chit[c] += 1
                        if a.get("has_mask") and a["segmentation"] and pm is not None and bi < len(pm):
                            gm = gtm(a["segmentation"], H, W)
                            if gm is not None and gm.sum():
                                q = pm[bi]
                                if q.shape != gm.shape:
                                    q = cv2.resize(q, (gm.shape[1], gm.shape[0]),
                                                   interpolation=cv2.INTER_NEAREST)
                                u = np.logical_or(q, gm).sum()
                                if u:
                                    mio[b].append(float(np.logical_and(q, gm).sum() / u))
                if n % 60 == 0:
                    log("  %s/%s %d/%d" % (name, role, n, len(by)))
            res[name][role] = {
                "by_size": {b: {"n": cnt[b], "hit": hit[b],
                                "mask_iou": (sorted(mio[b])[len(mio[b]) // 2] if mio[b] else None)}
                            for b in [bn(i) for i in range(len(B))] if cnt[bn(0)] or True},
                "by_class": {c: {"n": ccnt[c], "hit": chit[c]} for c in CL if ccnt[c]},
            }
            log("\n=== %s / %s (conf %.2f, IoU 0.5) ===" % (name, role, args.conf))
            log("%-8s %6s %6s %8s %10s" % ("크기", "정답", "찾음", "검출률", "마스크 IoU"))
            for i in range(len(B)):
                b = bn(i)
                if not cnt[b]:
                    continue
                q = res[name][role]["by_size"][b]["mask_iou"]
                log("%-8s %6d %6d %7.0f%% %10s" % (b, cnt[b], hit[b], 100 * hit[b] / cnt[b],
                                                   "%.3f" % q if q else "-"))
            t, h = sum(cnt.values()), sum(hit.values())
            log("%-8s %6d %6d %7.0f%%" % ("전체", t, h, 100 * h / max(1, t)))
        os.makedirs("outputs", exist_ok=True)
        json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    log("\n원자료: %s" % args.out)


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    main()
