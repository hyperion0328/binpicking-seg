#!/usr/bin/env python3
"""01 - 원본 4종을 학습 가능한 형태로 정규화한다.

입력  : data/by_resolution/<slug>/<WxH>/{images, annotations.coco.json}
출력  : data/build/<role>/{images, annotations.coco.json} + report.md

하는 일
  1. 쓰지 않는 그룹 제외 (주석 0개 / 인스턴스 극소)
  2. 자리표시자 카테고리 제거 + 클래스명 소문자 4종으로 통일
  3. 파일명 중복 제거 (같은 원본이 Roboflow 분할에 두 번 들어간 경우)
  4. rf-screw-nut-bolt 종횡비 복원 (512x512 stretch -> 512x288, 마스크까지)
  5. 역할 배정 : train/val/test · control(대조군) · crossdomain
  6. 촬영 세션 단위 연속 블록 분할 (인접 프레임이 학습/평가로 갈리지 않게)

주석은 COCO 형식 그대로 둔다. YOLO 변환은 SAM 으로 마스크를 채운 뒤 다음 단계에서.
ASCII 로그.
"""
import collections
import json
import os
import re
import shutil
import sys

import cv2
import numpy as np
from pycocotools import mask as maskutil

SRC = "data/by_resolution"
DST = "data/build"

CLASSES = ["bolt", "nut", "screw", "washer"]          # 최종 클래스 순서
DROP_GROUPS = {                                        # (slug, res) : 이유
    ("rf-arg-fixings3", "3072x4080"): "주석 0개",
    ("rf-arg-fixings3", "640x360"): "인스턴스 16개뿐, 4클래스 중 2종 결측",
}
CONTROL = ("rf-arg-fixings3", "640x480")               # 대조군: 학습 절대 금지
CROSSDOMAIN = ("rf-fasteners", "640x640")              # 교차 도메인 테스트
ASPECT_FIX = {("rf-screw-nut-bolt", "512x512"): 288}   # 목표 높이
SPLIT = (0.70, 0.15, 0.15)                             # train / val / test


def log(m):
    print(m, flush=True)


def base_name(fn):
    """Roboflow 해시를 떼어 원본 파일명을 얻는다."""
    return re.sub(r"[_.]rf\..*$", "", fn)


def session_key(slug, fn, cls):
    """같은 촬영 묶음이면 같은 문자열, 그 안의 순서는 정렬 가능한 값으로."""
    b = base_name(fn)
    if slug == "rf-screw-nut-bolt":
        m = re.match(r"snap_(\d+)", b)                  # 2.6시간 연속 촬영
        return ("snb:%s" % cls, int(m.group(1)) if m else 0)
    if slug == "rf-nuts-and-bolts":
        m = re.match(r"M6_frame_(\d+)", b)              # 동영상 프레임
        if m:
            return ("nb:M6", int(m.group(1)))
        m = re.match(r"nutsnbolts(\d+)", b)
        return ("nb:still", int(m.group(1)) if m else 0)
    if slug == "rf-arg-fixings3":
        m = re.match(r"([A-Za-z_]+)[-_]?(\d*)", b)      # Photo / nuts_bolts.. / PXL
        return ("af:%s" % (m.group(1) if m else b), m.group(2) or b if m else b)
    return ("%s:all" % slug, b)


def scale_ann(a, r, h_src, w_src):
    """세로만 r 배로 줄일 때 bbox·마스크를 함께 변환한다."""
    x, y, w, h = a["bbox"]
    a["bbox"] = [x, y * r, w, h * r]
    a["area"] = a.get("area", w * h) * r
    seg = a.get("segmentation")
    if isinstance(seg, list) and seg:                    # 폴리곤: y 좌표만
        a["segmentation"] = [[v * r if i % 2 else v for i, v in enumerate(p)]
                             for p in seg]
    elif isinstance(seg, dict):                          # RLE: 디코드->리사이즈->인코드
        s = dict(seg)
        if isinstance(s.get("counts"), str):
            s["counts"] = s["counts"].encode()
        m = maskutil.decode(s).astype(np.float32)
        m2 = cv2.resize(m, (w_src, int(round(h_src * r))), interpolation=cv2.INTER_AREA)
        m2 = (m2 > 0.5).astype(np.uint8)
        if m2.sum() == 0:                                # 너무 얇아 사라진 경우
            a["segmentation"] = []
            return False
        enc = maskutil.encode(np.asfortranarray(m2))
        enc["counts"] = enc["counts"].decode()
        a["segmentation"] = enc
    return True


def has_mask(a):
    s = a.get("segmentation")
    if isinstance(s, dict):
        return True
    return bool(s) and len(s[0]) // 2 > 4                 # 4점짜리는 사각형 취급


def main():
    if os.path.isdir(DST):
        shutil.rmtree(DST)

    groups = []
    for slug in sorted(os.listdir(SRC)):
        if not os.path.isdir(os.path.join(SRC, slug)):
            continue
        for res in sorted(os.listdir(os.path.join(SRC, slug))):
            p = os.path.join(SRC, slug, res, "annotations.coco.json")
            if os.path.exists(p):
                groups.append((slug, res, p))

    records = []          # dict(img, anns, role, session, order, src_path)
    stats = collections.Counter()
    dropped_masks = 0

    for slug, res, p in groups:
        if (slug, res) in DROP_GROUPS:
            d = json.load(open(p, encoding="utf-8"))
            log("  제외  %-22s %-11s %4d장 - %s"
                % (slug, res, len(d["images"]), DROP_GROUPS[(slug, res)]))
            stats["dropped_images"] += len(d["images"])
            continue

        d = json.load(open(p, encoding="utf-8"))
        cats = {c["id"]: c["name"].lower() for c in d["categories"]}
        by_img = collections.defaultdict(list)
        for a in d["annotations"]:
            by_img[a["image_id"]].append(a)

        role = ("control" if (slug, res) == CONTROL else
                "crossdomain" if (slug, res) == CROSSDOMAIN else "core")
        tgt_h = ASPECT_FIX.get((slug, res))
        seen = set()
        n_dup = 0

        for im in d["images"]:
            b = base_name(im["file_name"])
            if b in seen:                                 # 같은 원본 중복
                n_dup += 1
                stats["dup_images"] += 1
                continue
            seen.add(b)

            anns = []
            for a in by_img.get(im["id"], []):
                name = cats.get(a["category_id"])
                if name not in CLASSES:                   # 자리표시자 등
                    stats["dropped_anns"] += 1
                    continue
                a = dict(a)
                a["category_id"] = CLASSES.index(name)
                anns.append(a)

            w, h = im["width"], im["height"]
            if tgt_h:
                r = tgt_h / h
                for a in anns:
                    if not scale_ann(a, r, h, w):
                        dropped_masks += 1
                h = tgt_h

            cls = anns[0]["category_id"] if anns else -1
            sess, order = session_key(slug, im["file_name"],
                                      CLASSES[cls] if cls >= 0 else "none")
            records.append({
                "src": os.path.join(SRC, slug, res, "images", im["file_name"]),
                "file_name": im["file_name"], "width": w, "height": h,
                "anns": anns, "role": role, "session": sess, "order": order,
                "slug": slug, "res": res, "resize_h": tgt_h,
            })
        log("  사용  %-22s %-11s %4d장 (중복 %d 제외) -> %s"
            % (slug, res, len(seen), n_dup, role))

    # ---- 세션 단위 연속 블록 분할 ----
    core = [r for r in records if r["role"] == "core"]
    # 낱장 세션은 한 덩어리로 묶는다. 안 묶으면 n=1 세션이 전부 같은 쪽으로
    # 몰려(int(1*0.7)=0 -> 항상 test) 분할이 한쪽으로 기운다.
    cnt = collections.Counter(r["session"] for r in core)
    for r in core:
        if cnt[r["session"]] < 3:
            r["order"] = (r["session"], r["order"])
            r["session"] = "%s:misc" % r["slug"].replace("rf-", "")
    bysess = collections.defaultdict(list)
    for r in core:
        bysess[r["session"]].append(r)
    for sess, rs in bysess.items():
        rs.sort(key=lambda r: (str(r["order"]).zfill(12), r["file_name"]))
        n = len(rs)
        i1 = int(n * SPLIT[0])
        i2 = int(n * (SPLIT[0] + SPLIT[1]))
        for i, r in enumerate(rs):
            r["role"] = "train" if i < i1 else ("val" if i < i2 else "test")

    # ---- 출력 ----
    roles = collections.defaultdict(list)
    for r in records:
        roles[r["role"]].append(r)

    summary = []
    for role, rs in sorted(roles.items()):
        idir = os.path.join(DST, role, "images")
        os.makedirs(idir, exist_ok=True)
        images, annotations = [], []
        cls_cnt, mask_cnt = collections.Counter(), collections.Counter()
        for i, r in enumerate(rs, 1):
            dst = os.path.join(idir, r["file_name"])
            if r["resize_h"]:
                img = cv2.imread(r["src"], cv2.IMREAD_COLOR)
                img = cv2.resize(img, (r["width"], r["resize_h"]),
                                 interpolation=cv2.INTER_AREA)
                cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                shutil.copy2(r["src"], dst)
            images.append({"id": i, "file_name": r["file_name"],
                           "width": r["width"], "height": r["height"],
                           "source": r["slug"], "native_res": r["res"],
                           "session": r["session"]})
            for a in r["anns"]:
                a = dict(a, id=len(annotations) + 1, image_id=i)
                a["has_mask"] = has_mask(a)
                annotations.append(a)
                cls_cnt[CLASSES[a["category_id"]]] += 1
                if a["has_mask"]:
                    mask_cnt[CLASSES[a["category_id"]]] += 1
        json.dump({"images": images, "annotations": annotations,
                   "categories": [{"id": i, "name": c, "supercategory": "fastener"}
                                  for i, c in enumerate(CLASSES)]},
                  open(os.path.join(DST, role, "annotations.coco.json"), "w",
                       encoding="utf-8"), ensure_ascii=False)
        summary.append((role, len(images), len(annotations), cls_cnt, mask_cnt))
        log("  %-12s %5d장 %6d개" % (role, len(images), len(annotations)))

    # ---- 보고서 ----
    lines = ["# 01 전처리 결과", "",
             "`scripts/01_build_dataset.py` 산출. 원본은 `data/by_resolution/`.", "",
             "| 역할 | 사진 | 인스턴스 | bolt | nut | screw | washer | 마스크 보유 |",
             "|---|---|---|---|---|---|---|---|"]
    for role, ni, na, cc, mc in summary:
        mtot = sum(mc.values())
        lines.append("| %s | %d | %d | %d | %d | %d | %d | %d (%.1f%%) |"
                     % (role, ni, na, cc["bolt"], cc["nut"], cc["screw"],
                        cc["washer"], mtot, mtot / na * 100 if na else 0))
    lines += ["", "## 처리 내역", "",
              "- 제외 그룹: " + ", ".join("%s/%s (%s)" % (s, r, why)
                                        for (s, r), why in DROP_GROUPS.items()),
              "- 중복 파일명 제거: %d장" % stats["dup_images"],
              "- 자리표시자 등 버린 주석: %d개" % stats["dropped_anns"],
              "- 종횡비 복원: rf-screw-nut-bolt 512x512 -> 512x288 "
              "(세로 0.5625배, bbox·마스크 동시 변환)",
              "- 복원 중 사라진 마스크: %d개" % dropped_masks,
              "", "## 분할 규칙", "",
              "촬영 세션 단위로 정렬한 뒤 연속 블록으로 %d/%d/%d 나눴다. "
              "무작위로 나누면 1~2초 간격 연속 프레임이 학습과 평가로 갈려 "
              "실력이 아니라 기억력을 재게 된다." % tuple(int(x * 100) for x in SPLIT),
              ""]
    bys = collections.defaultdict(collections.Counter)
    for r in records:
        if r["role"] in ("train", "val", "test"):
            bys[r["session"]][r["role"]] += 1
    lines += ["| 세션 | train | val | test |", "|---|---|---|---|"]
    for s in sorted(bys):
        c = bys[s]
        lines.append("| `%s` | %d | %d | %d |" % (s, c["train"], c["val"], c["test"]))
    open(os.path.join(DST, "report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    log("\n  보고서: %s/report.md" % DST)


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    main()
