#!/usr/bin/env python3
"""04 - washer 의 낮은 IoU 를 두 원인으로 가르고, 모델을 바꾸면 나아지는지 본다.

타일 그림에서 두 가지가 보였다.
  (A) SAM 은 구멍을 제외한 도넛을 내는데 정답은 구멍까지 채운 원판이다.
      -> 물리적으로 SAM 이 옳지만 IoU 는 구멍 면적만큼 깎인다. 구멍을 메워 다시 재면 갈린다.
  (B) SAM 이 구멍만 잡는다. 이건 진짜 실패다. 예측 면적이 정답의 절반 미만인 경우로 센다.

세 모델을 같은 117개에 돌려 (A)(B) 비율과 IoU 를 비교한다.
ASCII 로그.
"""
import argparse, collections, json, math, os, time
import cv2, numpy as np
from pycocotools import mask as maskutil

CL = ["bolt", "nut", "screw", "washer"]
ROLES = ["train", "val", "test", "control"]


def gtm(s, h, w):
    if isinstance(s, dict):
        ss = dict(s)
        if isinstance(ss.get("counts"), str):
            ss["counts"] = ss["counts"].encode()
        return maskutil.decode(ss)
    return maskutil.decode(maskutil.merge(maskutil.frPyObjects(s, h, w)))


def fill_holes(m):
    """외곽선을 다시 채워 구멍을 메운다."""
    m = np.ascontiguousarray(m)          # pycocotools 는 Fortran 순서로 준다
    c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    o = np.zeros(m.shape, np.uint8)
    cv2.drawContours(o, c, -1, 1, -1)
    return o


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def load(cls="washer"):
    by = collections.defaultdict(list)
    for role in ROLES:
        p = "data/build/%s/annotations.coco.json" % role
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        im = {i["id"]: i for i in d["images"]}
        for a in d["annotations"]:
            if CL[a["category_id"]] != cls or not a.get("has_mask"):
                continue
            info = im[a["image_id"]]
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            by["data/build/%s/images/%s" % (role, info["file_name"])].append(
                {"bbox": [x, y, x + w, y + h], "seg": a["segmentation"],
                 "src": info.get("source", "?"), "size": math.sqrt(w * h)})
    return by


def run(model_name, by):
    from ultralytics import SAM
    model = SAM(model_name)
    rows = []
    for p, its in sorted(by.items()):
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        try:
            r = model(p, bboxes=[i["bbox"] for i in its], verbose=False)
        except Exception as e:
            print("  실패 %s : %s" % (os.path.basename(p), str(e)[:70]), flush=True)
            continue
        if r[0].masks is None:
            continue
        pred = r[0].masks.data.cpu().numpy().astype(np.uint8)
        for k, it in enumerate(its):
            if k >= len(pred):
                break
            g = gtm(it["seg"], h, w)
            if g.sum() == 0:
                continue
            pm = pred[k]
            if pm.shape != g.shape:
                pm = cv2.resize(pm, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
            pf, gf = fill_holes(pm), fill_holes(g)
            i_ = float(np.logical_and(pf, gf).sum())
            rows.append({"src": it["src"], "size": it["size"],
                         "iou": iou(pm, g), "iou_fill": iou(pf, gf),
                         "prec": i_ / pf.sum() if pf.sum() else 0.0,
                         "rec": i_ / gf.sum() if gf.sum() else 0.0,
                         "ratio": float(pm.sum() / g.sum()),
                         "gt_holed": int(gf.sum() > g.sum() * 1.02),
                         "pr_holed": int(pf.sum() > pm.sum() * 1.02)})
    return rows


def report(tag, rows):
    def agg(sel, lab):
        if not sel:
            return
        med = lambda k: sorted(x[k] for x in sel)[len(sel) // 2]
        print("%-22s %4d %7.3f %7.3f %7.3f %7.3f %6.0f%% %6.0f%% %6.0f%%" % (
            lab, len(sel), med("iou"), med("iou_fill"), med("prec"), med("rec"),
            100 * sum(1 for x in sel if x["iou_fill"] > .7) / len(sel),
            100 * sum(1 for x in sel if x["ratio"] < .5) / len(sel),
            100 * sum(x["pr_holed"] for x in sel) / len(sel)))
    print("\n=== washer / %s ===" % tag)
    print("%-22s %4s %7s %7s %7s %7s %6s %6s %6s" % (
        "", "n", "IoU", "구멍메움", "정밀도", "재현율", ">0.7", "붕괴", "SAM도넛"))
    for s in sorted({r["src"] for r in rows}):
        agg([r for r in rows if r["src"] == s], "  " + s)
    agg(rows, "  전체")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="sam_b.pt,sam2.1_b.pt,sam_l.pt")
    args = ap.parse_args()
    by = load("washer")
    print("washer 마스크 %d개 / 이미지 %d장" % (sum(len(v) for v in by.values()), len(by)), flush=True)
    out = {}
    for m in args.models.split(","):
        t = time.time()
        rows = run(m.strip(), by)
        out[m.strip()] = rows
        report("%s (%.0fs)" % (m.strip(), time.time() - t), rows)
    os.makedirs("outputs", exist_ok=True)
    json.dump(out, open("outputs/d_sam_washer.json", "w", encoding="utf-8"), ensure_ascii=False)
    print("\n원자료: outputs/d_sam_washer.json")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    main()
