#!/usr/bin/env python3
"""08 - 평가. 합산 하나로 보고하지 않는다.

E2 에서 배운 것 — 합친 평균은 출처 섞임을 가린다. 같은 imgsz 라도 소스마다 원본 해상도가
달라 실측 객체 크기가 22.1px ~ 56.0px 로 퍼진다. 그래서 **소스별로 따로 낸다.**

내는 지표(기획서 D14 · D16 · D18)
  - box AP  : 전량(`*_box`). washer 도 여기서만 잴 수 있다.
  - mask AP : 사람이 그린 윤곽이 전 인스턴스에 있는 사진만(`*_seg`). 주 지표.
  - 참고치  : SAM 채움본을 정답으로 쓴 `*_ref`. **주 지표 아님** - 정답이 모델 생성물이면
              기준자가 아니다. washer mask 는 여기서만 나온다(test_seg 에 5개뿐이라).
  - 대조군  : box AP 만(사람 윤곽 전량인 사진이 0장). 참고치를 병기.

평가 imgsz 는 **학습 imgsz 와 같게** 맞춘다(고정 조건).

**마스크 채점 해상도를 통일한다.** ultralytics 기본 경로는 마스크를 원형 해상도(imgsz/4)에서
채점하는데, P2 는 원형이 imgsz/2 라 잣대가 달라진다. `save_txt=True` 를 주면
`process_mask_native` 로 바뀌어 **입력 해상도 그대로** 채점하므로 두 구조가 같은 자를 쓴다.
P2 모델은 검증기도 보정본을 써야 한다(sodseg/p2_seg.py).
ASCII 로그.
"""
import argparse
import collections
import sys
import glob
import json
import os
import re

CL = ["bolt", "nut", "screw", "washer"]


def log(m):
    print(m, flush=True)


def imgsz_of(run_dir):
    """런 이름 끝의 숫자가 학습 imgsz 다. args.yaml 이 있으면 그쪽을 믿는다."""
    y = os.path.join(run_dir, "args.yaml")
    if os.path.exists(y):
        for line in open(y, encoding="utf-8"):
            m = re.match(r"\s*imgsz:\s*(\d+)", line)
            if m:
                return int(m.group(1))
    m = re.search(r"_(\d+)$", os.path.basename(run_dir))
    return int(m.group(1)) if m else 640


def source_lists(base_list, out_dir):
    """평가 목록을 출처별로 쪼갠 목록 파일을 만든다."""
    paths = [l for l in open(base_list).read().split("\n") if l.strip()]
    # 경로에서 출처를 알 수 없으므로 build 의 주석에서 파일명 -> 출처 표를 만든다
    src = {}
    for j in glob.glob("data/build/*/annotations*.coco.json"):
        d = json.load(open(j, encoding="utf-8"))
        for i in d["images"]:
            src.setdefault(i["file_name"], i.get("source", "?"))
    by = collections.defaultdict(list)
    for p in paths:
        by[src.get(os.path.basename(p), "?")].append(p)
    made = {}
    os.makedirs(out_dir, exist_ok=True)
    for s, items in sorted(by.items()):
        if len(items) < 5:
            continue
        lp = os.path.join(out_dir, "%s__%s.txt" % (
            os.path.splitext(os.path.basename(base_list))[0], s))
        open(lp, "w").write("\n".join(items) + "\n")
        made[s] = (lp, len(items))
    return made


def write_yaml(path, val_list):
    open(path, "w").write(
        "path: %s\ntrain: %s\nval: %s\nnames:\n" % (
            os.path.abspath("data/yolo"), os.path.abspath(val_list), os.path.abspath(val_list))
        + "\n".join("  %d: %s" % (i, c) for i, c in enumerate(CL)) + "\n")
    return path


def run_val(weights, yml, imgsz, split_name):
    from ultralytics import YOLO
    from sodseg.p2_seg import P2SegmentationValidator
    m = YOLO(weights)
    r = m.val(validator=P2SegmentationValidator,      # 기본 헤드에서도 동작이 같다
              # 평가 배치는 지표에 영향이 없다(전체 사진에 대해 AP 를 낸다).
              # 통일 잣대는 마스크를 입력 해상도 그대로 만들어 batch 16 x 300검출 x 640^2 가
              # m 모델에서 14GB 를 요구해 죽었다. 4 로 낮춘다.
              data=yml, imgsz=imgsz, batch=4, device="0", verbose=False, plots=False,
              save_txt=True,                          # -> process_mask_native: 잣대 통일
              project=os.path.abspath("runs/val"), name=split_name, exist_ok=True)
    out = {"box_mAP50_95": float(r.box.map), "box_mAP50": float(r.box.map50),
           "mask_mAP50_95": float(r.seg.map), "mask_mAP50": float(r.seg.map50),
           "per_class": {}}
    for i, c in enumerate(CL):
        idx = list(r.box.ap_class_index)
        if i in idx:
            k = idx.index(i)
            out["per_class"][c] = {"box": float(r.box.maps[i]), "mask": float(r.seg.maps[i]),
                                   "n": int(r.box.nt_per_class[i]) if hasattr(r.box, "nt_per_class") else -1}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/seg")
    ap.add_argument("--out", default="outputs/d_eval.json")
    ap.add_argument("--only", default="", help="쉼표 구분 런 이름. 주면 그것만 재고 기존 결과에 합친다")
    args = ap.parse_args()

    tmp = "data/yolo/bysrc"
    targets = [
        ("test_seg", "data/yolo/test_seg.txt", "mask"),     # 주 지표(mask)
        ("test_box", "data/yolo/test_box.txt", "box"),      # 주 지표(box)
        ("control", "data/yolo/control_box.txt", "box"),    # 대조군 - box 만 (D14)
        ("crossdomain", "data/yolo/crossdomain_seg.txt", "both"),
        ("test_ref", "data/yolo/test_ref.txt", "ref"),      # 참고치
        ("control_ref", "data/yolo/control_ref.txt", "ref"),
    ]
    # 이미 잰 것을 지우지 않는다 - 한 런이 죽어도 나머지를 다시 재지 않아도 되게.
    res = json.load(open(args.out, encoding="utf-8")) if os.path.exists(args.out) else {}
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    for run in sorted(glob.glob(os.path.join(args.runs, "*"))):
        if only and os.path.basename(run) not in only:
            continue
        w = os.path.join(run, "weights", "best.pt")
        if not os.path.exists(w):
            continue
        name = os.path.basename(run)
        sz = imgsz_of(run)
        log("\n########## %s (imgsz %d) ##########" % (name, sz))
        res[name] = {"imgsz": sz, "splits": {}}
        for tag, lst, kind in targets:
            if not os.path.exists(lst) or not open(lst).read().strip():
                continue
            y = write_yaml("data/yolo/eval_%s.yaml" % tag, lst)
            log("  [%s] 전체" % tag)
            res[name]["splits"][tag] = {"all": run_val(w, y, sz, "%s_%s" % (name, tag))}
            for s, (lp, n) in source_lists(lst, tmp).items():
                ys = write_yaml("data/yolo/eval_%s__%s.yaml" % (tag, s), lp)
                log("  [%s] %s (%d장)" % (tag, s, n))
                res[name]["splits"][tag][s] = run_val(w, ys, sz, "%s_%s_%s" % (name, tag, s))
        os.makedirs("outputs", exist_ok=True)
        json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    log("\n=== 요약: 주 지표 ===")
    log("%-20s %5s %10s %10s %10s" % ("런", "imgsz", "test box", "test mask", "대조군 box"))
    for k, v in sorted(res.items()):
        sp = v["splits"]
        log("%-20s %5d %10.4f %10.4f %10.4f" % (
            k, v["imgsz"],
            sp.get("test_box", {}).get("all", {}).get("box_mAP50_95", float("nan")),
            sp.get("test_seg", {}).get("all", {}).get("mask_mAP50_95", float("nan")),
            sp.get("control", {}).get("all", {}).get("box_mAP50_95", float("nan"))))
    log("\n원자료: %s" % args.out)


if __name__ == "__main__":
    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    os.chdir(ROOT)
    sys.path.insert(0, os.path.abspath(ROOT))
    main()
