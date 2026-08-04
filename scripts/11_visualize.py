#!/usr/bin/env python3
"""11 - 모델이 실제로 무엇을 찾고 무엇을 놓치는지 그려 본다.

숫자 하나(mAP)로는 "어디서 무너지는가"가 안 보인다. 세 가지를 그린다.

  1) overlay  : 사진 한 장에 정답(초록)과 예측(빨강)을 겹쳐 그린다.
  2) miss     : 정답인데 못 찾은 것만 잘라 타일로. 대조군에서 이게 핵심 질문이다.
  3) compare  : 같은 사진을 여러 런으로 나란히. 조건 차이가 눈에 보인다.

ASCII 로그.
"""
import argparse
import collections
import json
import math
import os

import cv2
import numpy as np
from pycocotools import mask as maskutil

CL = ["bolt", "nut", "screw", "washer"]
COLOR = {"gt": (80, 220, 80), "pred": (60, 60, 245), "miss": (0, 200, 255)}


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


def load_gt(role):
    p = "data/build/%s/annotations.coco.json" % role
    d = json.load(open(p, encoding="utf-8"))
    im = {i["id"]: i for i in d["images"]}
    by = collections.defaultdict(list)
    for a in d["annotations"]:
        by[a["image_id"]].append(a)
    return im, by


def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / ua if ua > 0 else 0.0


def draw(img, boxes, color, labels=None, thick=2):
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        if labels:
            cv2.putText(img, labels[i], (x1, max(10, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/seg/yolo11s-seg_640")
    ap.add_argument("--role", default="control")
    ap.add_argument("--imgsz", type=int, default=0, help="0 이면 런 이름에서 뽑는다")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--mode", default="overlay", choices=["overlay", "miss", "compare"])
    ap.add_argument("--runs", default="", help="compare 용, 쉼표 구분")
    ap.add_argument("--out", default="outputs/viz")
    args = ap.parse_args()

    import re
    sz = args.imgsz or int(re.search(r"_(\d+)", os.path.basename(args.run)).group(1))
    from ultralytics import YOLO

    im, by = load_gt(args.role)
    ids = [i for i in sorted(im, key=lambda k: im[k]["file_name"]) if by.get(i)][:args.n]
    os.makedirs(args.out, exist_ok=True)

    runs = [r.strip() for r in args.runs.split(",") if r.strip()] or [args.run]
    models = {}
    for r in runs:
        models[r] = YOLO(os.path.join(r, "weights", "best.pt"))

    if args.mode == "miss":
        tiles = []
        m = models[runs[0]]
        for iid in ids or sorted(im):
            info = im[iid]
            fp = "data/build/%s/images/%s" % (args.role, info["file_name"])
            img = cv2.imread(fp)
            if img is None:
                continue
            res = m(fp, imgsz=sz, conf=args.conf, verbose=False)[0]
            pb = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
            for a in by[iid]:
                x, y, bw, bh = a["bbox"]
                g = [x, y, x + bw, y + bh]
                hit = any(iou_xyxy(g, p) >= 0.5 for p in pb)
                pad = max(10, int(0.9 * max(bw, bh)))
                x1, y1 = max(0, int(x - pad)), max(0, int(y - pad))
                x2, y2 = min(img.shape[1], int(x + bw + pad)), min(img.shape[0], int(y + bh + pad))
                c = img[y1:y2, x1:x2].copy()
                if c.size == 0:
                    continue
                cv2.rectangle(c, (int(x - x1), int(y - y1)),
                              (int(x + bw - x1), int(y + bh - y1)),
                              COLOR["gt"] if hit else COLOR["miss"], 1)
                c = cv2.resize(c, (128, 128), interpolation=cv2.INTER_NEAREST)
                cv2.putText(c, "%s %.0fpx %s" % (CL[a["category_id"]][:5],
                                                 math.sqrt(bw * bh), "O" if hit else "X"),
                            (3, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                            (80, 220, 80) if hit else (0, 200, 255), 1)
                tiles.append((hit, c))
        miss = [t[1] for t in tiles if not t[0]]
        hits = [t[1] for t in tiles if t[0]]
        log("정답 %d개 중 찾음 %d · 놓침 %d (conf %.2f, IoU 0.5)"
            % (len(tiles), len(hits), len(miss), args.conf))
        sel = miss[:60] or hits[:60]
        cols = 12
        rows = [np.hstack(sel[i:i + cols] + [np.zeros((128, 128, 3), np.uint8)]
                          * (cols - len(sel[i:i + cols]))) for i in range(0, len(sel), cols)]
        if rows:
            p = os.path.join(args.out, "miss_%s_%s.jpg"
                             % (os.path.basename(args.run), args.role))
            cv2.imwrite(p, np.vstack(rows))
            log("-> %s" % p)
        return

    for iid in ids:
        info = im[iid]
        fp = "data/build/%s/images/%s" % (args.role, info["file_name"])
        base = cv2.imread(fp)
        if base is None:
            continue
        h, w = base.shape[:2]
        panels = []
        # 정답 패널
        gtp = base.copy()
        for a in by[iid]:
            x, y, bw, bh = a["bbox"]
            g = gtm(a["segmentation"], h, w) if a.get("has_mask") else None
            if g is not None:
                cv2.drawContours(gtp, cv2.findContours(np.ascontiguousarray(g), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)[0], -1, COLOR["gt"], 2)
            draw(gtp, [[x, y, x + bw, y + bh]], COLOR["gt"], [CL[a["category_id"]]], 1)
        cv2.putText(gtp, "GT  (%d)" % len(by[iid]), (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR["gt"], 2)
        panels.append(gtp)

        for r in runs:
            res = models[r](fp, imgsz=sz, conf=args.conf, verbose=False)[0]
            p = base.copy()
            n = 0
            if res.masks is not None:
                mk = res.masks.data.cpu().numpy().astype(np.uint8)
                for k in range(len(mk)):
                    mm = cv2.resize(mk[k], (w, h), interpolation=cv2.INTER_NEAREST)
                    cv2.drawContours(p, cv2.findContours(np.ascontiguousarray(mm),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                                     -1, COLOR["pred"], 2)
            if res.boxes is not None:
                bx = res.boxes.xyxy.cpu().numpy()
                cf = res.boxes.conf.cpu().numpy()
                cs = res.boxes.cls.cpu().numpy().astype(int)
                n = len(bx)
                draw(p, bx, COLOR["pred"],
                     ["%s %.2f" % (CL[c][:5], f) for c, f in zip(cs, cf)], 1)
            cv2.putText(p, "%s  (%d)" % (os.path.basename(r), n), (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR["pred"], 2)
            panels.append(p)

        out = np.hstack(panels)
        sc = 1900.0 / out.shape[1]
        if sc < 1:
            out = cv2.resize(out, (int(out.shape[1] * sc), int(out.shape[0] * sc)))
        p = os.path.join(args.out, "%s_%s_%s.jpg"
                         % (args.mode, args.role, os.path.splitext(info["file_name"])[0][:28]))
        cv2.imwrite(p, out)
        log("-> %s" % p)


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    main()
