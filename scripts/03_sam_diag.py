#!/usr/bin/env python3
"""03 - SAM 의 낮은 IoU 가 SAM 탓인지 정답 마스크 탓인지 가른다.

IoU 하나로는 방향을 모른다. 같은 인스턴스에서
  - precision = |pred & gt| / |pred|   (SAM 이 정답 밖으로 샜는가)
  - recall    = |pred & gt| / |gt|     (SAM 이 정답을 덜 덮었는가)
  - 면적비    = |pred| / |gt|
  - 정답 폴리곤 꼭짓점 수              (정답이 거친가)
를 함께 재면 갈린다. precision 높고 recall 낮으면 정답이 부풀어 있는 것이고,
반대면 SAM 이 이웃 부품까지 삼킨 것이다.

ASCII 로그.
"""
import argparse, collections, json, math, os, time
import cv2, numpy as np
from pycocotools import mask as maskutil

CL = ["bolt", "nut", "screw", "washer"]
ROLES = ["train", "val", "test", "control"]


def gt_mask(s, h, w):
    if isinstance(s, dict):
        ss = dict(s)
        if isinstance(ss.get("counts"), str):
            ss["counts"] = ss["counts"].encode()
        return maskutil.decode(ss)
    if isinstance(s, list) and s:
        return maskutil.decode(maskutil.merge(maskutil.frPyObjects(s, h, w)))
    return None


def nverts(s):
    if isinstance(s, list):
        return sum(len(p) // 2 for p in s)
    return -1                                    # RLE 는 꼭짓점 개념이 없다


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sam_b.pt")
    ap.add_argument("--source", default="rf-arg-fixings3")
    ap.add_argument("--dump", type=int, default=12, help="겹쳐 그린 그림 저장 장수")
    args = ap.parse_args()
    from ultralytics import SAM

    byimg = collections.defaultdict(list)
    for role in ROLES:
        p = "data/build/%s/annotations.coco.json" % role
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        im = {i["id"]: i for i in d["images"]}
        for a in d["annotations"]:
            if not a.get("has_mask"):
                continue
            info = im[a["image_id"]]
            if args.source and info.get("source") != args.source:
                continue
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            byimg["data/build/%s/images/%s" % (role, info["file_name"])].append({
                "bbox": [x, y, x + w, y + h], "size": math.sqrt(w * h),
                "cls": a["category_id"], "seg": a["segmentation"],
                "nv": nverts(a["segmentation"]), "role": role})

    model = SAM(args.model)
    paths = sorted(byimg)
    print("%s : 이미지 %d장 / 인스턴스 %d개" % (args.source, len(paths),
                                              sum(len(v) for v in byimg.values())), flush=True)
    os.makedirs("outputs/sam_diag", exist_ok=True)
    rows, dumped, t0 = [], 0, time.time()
    for n, p in enumerate(paths, 1):
        items = byimg[p]
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        try:
            res = model(p, bboxes=[it["bbox"] for it in items], verbose=False)
        except Exception as e:
            print("  실패 %s : %s" % (os.path.basename(p), str(e)[:80]), flush=True)
            continue
        if res[0].masks is None:
            continue
        pred = res[0].masks.data.cpu().numpy().astype(np.uint8)
        vis = img.copy() if dumped < args.dump else None
        for k, it in enumerate(items):
            if k >= len(pred):
                break
            g = gt_mask(it["seg"], h, w)
            if g is None or g.sum() == 0:
                continue
            pm = pred[k]
            if pm.shape != g.shape:
                pm = cv2.resize(pm, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
            i_ = float(np.logical_and(pm, g).sum())
            pa, ga = float(pm.sum()), float(g.sum())
            u_ = pa + ga - i_
            rows.append({"cls": CL[it["cls"]], "size": it["size"], "nv": it["nv"],
                         "iou": i_ / u_ if u_ else 0.0,
                         "prec": i_ / pa if pa else 0.0, "rec": i_ / ga if ga else 0.0,
                         "ratio": pa / ga if ga else 0.0, "role": it["role"]})
            if vis is not None:
                cv2.drawContours(vis, cv2.findContours(g, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 0), 2)
                cv2.drawContours(vis, cv2.findContours(pm, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 0, 255), 2)
        if vis is not None:
            cv2.imwrite("outputs/sam_diag/%s" % os.path.basename(p).replace(".jpg", "_gt-green_sam-red.jpg"), vis)
            dumped += 1
        if n % 40 == 0:
            print("  %d/%d · %d개 · %.0fs" % (n, len(paths), len(rows), time.time() - t0), flush=True)

    with open("outputs/d_sam_diag_%s.jsonl" % args.source, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def agg(sel, lab):
        if not sel:
            return
        m = lambda k: sorted(x[k] for x in sel)[len(sel) // 2]
        print("%-14s %5d %7.3f %7.3f %7.3f %7.3f %7.0f" % (
            lab, len(sel), m("iou"), m("prec"), m("rec"), m("ratio"), m("nv")))
    print("\n=== %s · 중앙값 ===" % args.source)
    print("%-14s %5s %7s %7s %7s %7s %7s" % ("", "n", "IoU", "정밀도", "재현율", "면적비", "꼭짓점"))
    for c in CL:
        agg([r for r in rows if r["cls"] == c], c)
    agg(rows, "전체")
    print("\n원자료: outputs/d_sam_diag_%s.jsonl · 그림 outputs/sam_diag/ (%d장)" % (args.source, dumped))


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    main()
