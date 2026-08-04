#!/usr/bin/env python3
"""05 - 검증을 통과한 SAM 으로 빠진 마스크를 채운다.

03/04 의 실측이 근거다.
  - SAM 은 정답 안쪽에 더 좁게 들어간다(정밀도 0.98 / 재현율 0.70). 밖으로 새지 않는다.
  - 정답 washer 마스크는 100% 가 구멍을 메운 원판이다. SAM 도넛을 그대로 넣으면
    같은 클래스에 두 관례가 섞이므로 구멍을 메워 맞춘다.
  - sam_b 가 sam2.1_b·sam_l 보다 정밀한 정답에 대해 가장 조인다.

채우는 곳은 **train 뿐이다.** val/test/control 을 모델 생성물로 채우면 채점표를
채점 대상이 만든 꼴이 된다(D14 와 같은 이유). 평가는 따로 만든다.

관문을 통과 못 한 마스크는 채우지 않고 `has_mask=False` 로 남긴다 — 왜 못 채웠는지도 센다.
ASCII 로그.
"""
import argparse
import collections
import json
import math
import os
import time

import cv2
import numpy as np
from pycocotools import mask as maskutil

CL = ["bolt", "nut", "screw", "washer"]

# 관문 - 03/04 에서 본 실패 유형만 막는다. 통과율을 먼저 보고 조인다.
MIN_FILL = 0.15      # 마스크 면적 / 박스 면적. 이보다 작으면 구멍만 잡은 붕괴로 본다
MAX_FILL = 1.05      # 박스 안에 들어와야 한다
MIN_MAIN = 0.80      # 가장 큰 덩어리가 이 비율 미만이면 여러 물체를 삼킨 것


def log(m):
    print(m, flush=True)


def fill_holes(m):
    m = np.ascontiguousarray(m)
    c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    o = np.zeros(m.shape, np.uint8)
    cv2.drawContours(o, c, -1, 1, -1)
    return o


def gate(pm, bbox):
    """(통과여부, 사유, 정리된 마스크)"""
    x, y, w, h = bbox
    if pm.sum() == 0:
        return False, "빈 마스크", None
    n, lab, stats, _ = cv2.connectedComponentsWithStats(np.ascontiguousarray(pm), 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        big = int(np.argmax(areas)) + 1
        if areas.max() / areas.sum() < MIN_MAIN:
            return False, "덩어리 분산", None
        pm = (lab == big).astype(np.uint8)      # 잔부스러기 제거
    pm = fill_holes(pm)
    r = pm.sum() / max(1.0, w * h)
    if r < MIN_FILL:
        return False, "면적 붕괴", None
    if r > MAX_FILL:
        return False, "박스 초과", None
    # 박스 밖으로 나간 화소는 잘라낸다(SAM 이 이웃을 살짝 물었을 때)
    keep = np.zeros(pm.shape, np.uint8)
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(pm.shape[1], int(math.ceil(x + w))), min(pm.shape[0], int(math.ceil(y + h)))
    keep[y0:y1, x0:x1] = 1
    pm = pm * keep
    if pm.sum() / max(1.0, w * h) < MIN_FILL:
        return False, "박스밖 절단후 붕괴", None
    return True, "", pm


def to_poly(m):
    """마스크 -> COCO 폴리곤. 가장 큰 외곽선 하나만 쓴다."""
    c, _ = cv2.findContours(np.ascontiguousarray(m), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    if not c:
        return None
    c = max(c, key=cv2.contourArea)
    eps = 0.004 * cv2.arcLength(c, True)          # 과하게 줄이면 모양이 뭉개진다
    ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    if len(ap) < 3:
        return None
    return [[float(v) for xy in ap for v in xy]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sam_b.pt")
    ap.add_argument("--roles", default="train")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    from ultralytics import SAM

    model = SAM(args.model)
    total = collections.Counter()
    reasons = collections.Counter()
    per_cls = collections.defaultdict(lambda: [0, 0])

    for role in args.roles.split(","):
        role = role.strip()
        src = "data/build/%s/annotations.coco.json" % role
        d = json.load(open(src, encoding="utf-8"))
        im = {i["id"]: i for i in d["images"]}
        byimg = collections.defaultdict(list)
        for a in d["annotations"]:
            if not a.get("has_mask"):
                byimg[a["image_id"]].append(a)
        log("[%s] 채울 인스턴스 %d개 / 사진 %d장"
            % (role, sum(len(v) for v in byimg.values()), len(byimg)))

        t0 = time.time()
        for n, (iid, anns) in enumerate(sorted(byimg.items()), 1):
            info = im[iid]
            p = "data/build/%s/images/%s" % (role, info["file_name"])
            img = cv2.imread(p)
            if img is None:
                reasons["사진 못 읽음"] += len(anns)
                continue
            h, w = img.shape[:2]
            boxes = [[a["bbox"][0], a["bbox"][1],
                      a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns]
            try:
                r = model(p, bboxes=boxes, verbose=False)
            except Exception as e:
                reasons["SAM 예외"] += len(anns)
                log("  예외 %s : %s" % (info["file_name"][:40], str(e)[:70]))
                continue
            if r[0].masks is None:
                reasons["마스크 없음"] += len(anns)
                continue
            pred = r[0].masks.data.cpu().numpy().astype(np.uint8)
            for k, a in enumerate(anns):
                total["시도"] += 1
                per_cls[CL[a["category_id"]]][1] += 1
                if k >= len(pred):
                    reasons["예측 부족"] += 1
                    continue
                pm = pred[k]
                if pm.shape != (h, w):
                    pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST)
                ok, why, cleaned = gate(pm, a["bbox"])
                if not ok:
                    reasons[why] += 1
                    continue
                poly = to_poly(cleaned)
                if poly is None:
                    reasons["폴리곤 실패"] += 1
                    continue
                a["segmentation"] = poly
                a["has_mask"] = True
                a["pseudo"] = True                 # 지웠다 되돌릴 수 있게 표시
                a["pseudo_model"] = args.model
                a["area"] = float(cleaned.sum())
                total["통과"] += 1
                per_cls[CL[a["category_id"]]][0] += 1
            if n % 20 == 0:
                log("  %d/%d 사진 · 통과 %d · %.0fs"
                    % (n, len(byimg), total["통과"], time.time() - t0))

        if not args.dry_run:
            out = "data/build/%s/annotations.filled.coco.json" % role
            json.dump(d, open(out, "w", encoding="utf-8"), ensure_ascii=False)
            log("  -> %s" % out)

    log("\n=== 채우기 결과 (%s) ===" % args.model)
    log("시도 %d · 통과 %d (%.1f%%)"
        % (total["시도"], total["통과"], 100 * total["통과"] / max(1, total["시도"])))
    log("%-8s %8s %8s %7s" % ("클래스", "통과", "시도", "통과율"))
    for c in CL:
        ok, tr = per_cls[c]
        if tr:
            log("%-8s %8d %8d %6.1f%%" % (c, ok, tr, 100 * ok / tr))
    if reasons:
        log("\n못 채운 사유")
        for k, v in reasons.most_common():
            log("  %-20s %5d" % (k, v))


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    main()
