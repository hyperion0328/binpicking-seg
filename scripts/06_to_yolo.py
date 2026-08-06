#!/usr/bin/env python3
"""06 - COCO -> YOLO-seg 라벨. 이미지는 복사하지 않는다.

ultralytics 는 이미지 경로의 마지막 `/images/` 를 `/labels/` 로 바꿔 라벨을 찾는다.
`data/build/<역할>/images/x.jpg` 옆에 `.../labels/x.txt` 를 놓으면 그대로 맞는다.
학습은 epoch 마다 이미지를 전부 다시 읽으므로 복제본을 만들지 않는다.

라벨 규칙 — 인스턴스마다 폴리곤 한 줄.
  - 마스크가 있으면 그 외곽선(RLE 는 화소에서 외곽선을 뽑는다).
  - 마스크가 없으면 **박스를 폴리곤으로** 넣는다. 마스크는 부정확하지만 박스는 정답이고,
    지우면 그 물체가 배경으로 학습되어 오탐 억제를 배운다. 대신 그 사진은
    mask AP 목록에서 뺀다.

평가 목록을 셋으로 나눈다.
  - `*_seg.txt` : **전 인스턴스에 사람이 그린 마스크가 있는 사진만.** mask AP 주 지표는 여기서만.
  - `*_box.txt` : 전량. box AP 는 박스 폴리곤에서도 정확히 복원되므로 문제없다.
  - `*_ref.txt` : SAM 으로 채운 마스크까지 정답으로 쓴 **참고용**(D14). washer 는 마스크의 96% 가
    결측이라 사람 마스크만 남기면 test 에 5개밖에 안 남아 mask AP 를 낼 수 없다. 이 목록의 성적은
    **참고치로만 병기**하고 주 지표로 쓰지 않는다 - 정답이 모델 생성물이면 기준자가 아니다.
    라벨은 `<역할>/labels_ref/` 에 두고 `<역할>_ref/` 심볼릭 링크로 가리킨다(사진 복제 없음).

ASCII 로그.
"""
import argparse
import collections
import json
import os

import cv2
import numpy as np
from pycocotools import mask as maskutil
from ultralytics.data.converter import merge_multi_segment

CL = ["bolt", "nut", "screw", "washer"]
ROLES = ["train", "val", "test", "control", "crossdomain"]
MAX_PTS = 120          # 폴리곤 점 수 상한. 넘으면 단순화한다
STAT = collections.Counter()


def log(m):
    print(m, flush=True)


def decode(seg, h, w):
    if isinstance(seg, dict):
        s = dict(seg)
        if isinstance(s.get("counts"), str):
            s["counts"] = s["counts"].encode()
        return maskutil.decode(s)
    return maskutil.decode(maskutil.merge(maskutil.frPyObjects(seg, h, w)))


def to_points(seg, h, w):
    """마스크를 (N,2) 폴리곤 하나로. YOLO-seg 는 인스턴스당 폴리곤 하나만 받는다.

    가려져서 조각난 마스크에서 가장 큰 조각만 쓰면 물체가 잘린다 —
    실측으로 1.8% 가 조각나 있었고 심하면 박스 범위의 90% 가 날아갔다.
    ultralytics 의 merge_multi_segment 로 조각을 이어 붙인다.
    """
    if isinstance(seg, list) and len(seg) == 1 and len(seg[0]) >= 6:
        p = np.asarray(seg[0], np.float32).reshape(-1, 2)      # 폴리곤은 그대로 쓴다
    else:
        m = np.ascontiguousarray(decode(seg, h, w))
        c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = [x for x in c if len(x) >= 3 and cv2.contourArea(x) > 0]
        if not c:
            return None
        if len(c) == 1:
            p = c[0].reshape(-1, 2).astype(np.float32)
        else:
            segs = [x.reshape(-1).astype(np.float32) for x in
                    sorted(c, key=cv2.contourArea, reverse=True)]
            try:
                p = np.concatenate(merge_multi_segment(segs)).reshape(-1, 2).astype(np.float32)
            except Exception:
                p = max(c, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
                STAT["병합 실패"] += 1
    if len(p) > MAX_PTS:
        # 조각을 이은 폴리곤은 되짚는 선을 지나므로 단순화가 그 다리를 지울 수 있다.
        # 점 수만 줄이도록 아주 작은 오차만 허용한다.
        eps = 0.0008 * cv2.arcLength(p.reshape(-1, 1, 2).astype(np.float32), True)
        q = cv2.approxPolyDP(p.reshape(-1, 1, 2).astype(np.float32), eps, True).reshape(-1, 2)
        if len(q) >= 3:
            p = q
    return p if len(p) >= 3 else None


def box_points(b):
    x, y, w, h = b
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/yolo")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    stat = STAT
    lists = {}

    for role in ROLES:
        base = "data/build/%s" % role
        fn = "annotations.filled.coco.json" if role == "train" else "annotations.coco.json"
        src = os.path.join(base, fn)
        if not os.path.exists(src):
            continue
        d = json.load(open(src, encoding="utf-8"))
        im = {i["id"]: i for i in d["images"]}
        byimg = collections.defaultdict(list)
        for a in d["annotations"]:
            byimg[a["image_id"]].append(a)

        ldir = os.path.join(base, "labels")
        os.makedirs(ldir, exist_ok=True)
        seg_ok, all_img = [], []
        for iid, info in sorted(im.items(), key=lambda kv: kv[1]["file_name"]):
            ip = os.path.abspath(os.path.join(base, "images", info["file_name"]))
            if not os.path.exists(ip):
                stat["사진 없음"] += 1
                continue
            h, w = info["height"], info["width"]
            lines, human = [], True
            anns = byimg.get(iid, [])
            for a in anns:
                pts = None
                if a.get("has_mask"):
                    pts = to_points(a["segmentation"], h, w)
                    if a.get("pseudo"):
                        human = False              # SAM 이 만든 것은 사람 정답이 아니다
                if pts is None:
                    pts = box_points(a["bbox"])    # 관문 탈락 · 마스크 없음 -> 박스
                    human = False
                    stat["%s:박스 대체" % role] += 1
                p = pts.copy()
                p[:, 0] = np.clip(p[:, 0] / w, 0, 1)
                p[:, 1] = np.clip(p[:, 1] / h, 0, 1)
                lines.append("%d " % a["category_id"] +
                             " ".join("%.6f" % v for v in p.reshape(-1)))
                stat["%s:인스턴스" % role] += 1
            txt = os.path.join(ldir, os.path.splitext(info["file_name"])[0] + ".txt")
            open(txt, "w").write("\n".join(lines) + ("\n" if lines else ""))
            all_img.append(ip)
            if anns and human:
                seg_ok.append(ip)
            stat["%s:사진" % role] += 1

        # 참고용 - SAM 채움본까지 정답으로 쓴 라벨. 주 지표 아님(D14).
        ref_img = []
        fj = os.path.join(base, "annotations.filled.coco.json")
        if role != "train" and os.path.exists(fj):
            fd = json.load(open(fj, encoding="utf-8"))
            fim = {i["id"]: i for i in fd["images"]}
            fby = collections.defaultdict(list)
            for a in fd["annotations"]:
                fby[a["image_id"]].append(a)
            rdir = os.path.join(base, "labels_ref")
            os.makedirs(rdir, exist_ok=True)
            link = os.path.abspath("data/build/%s_ref" % role)
            os.makedirs(link, exist_ok=True)
            for a, b in [("images", os.path.join(base, "images")), ("labels", rdir)]:
                lp = os.path.join(link, a)
                if os.path.islink(lp) or os.path.exists(lp):
                    if os.path.islink(lp):
                        os.unlink(lp)
                    else:
                        continue
                os.symlink(os.path.abspath(b), lp)
            for iid, info in sorted(fim.items(), key=lambda kv: kv[1]["file_name"]):
                ip = os.path.join(base, "images", info["file_name"])
                if not os.path.exists(ip):
                    continue
                h, w = info["height"], info["width"]
                lines = []
                for a in fby.get(iid, []):
                    pts = to_points(a["segmentation"], h, w) if (
                        a.get("has_mask") and a["segmentation"]) else None
                    if pts is None:
                        pts = box_points(a["bbox"])
                    p = pts.copy()
                    p[:, 0] = np.clip(p[:, 0] / w, 0, 1)
                    p[:, 1] = np.clip(p[:, 1] / h, 0, 1)
                    lines.append("%d " % a["category_id"] +
                                 " ".join("%.6f" % v for v in p.reshape(-1)))
                open(os.path.join(rdir, os.path.splitext(info["file_name"])[0] + ".txt"),
                     "w").write("\n".join(lines) + ("\n" if lines else ""))
                ref_img.append(os.path.join(link, "images", info["file_name"]))

        for tag, items in [("box", all_img), ("seg", seg_ok), ("ref", ref_img)]:
            lp = os.path.join(args.out, "%s_%s.txt" % (role, tag))
            open(lp, "w").write("\n".join(items) + "\n")
            lists["%s_%s" % (role, tag)] = (lp, len(items))
        log("[%-11s] 사진 %4d · 인스턴스 %5d · 사람마스크 전량인 사진 %4d"
            % (role, len(all_img), stat["%s:인스턴스" % role], len(seg_ok)))

    # data yaml - 학습용과 평가용을 나눠 쓴다
    root = os.path.abspath(args.out)
    names = "\n".join("  %d: %s" % (i, c) for i, c in enumerate(CL))
    def yml(name, val_list):
        p = os.path.join(args.out, name)
        open(p, "w").write(
            "# scripts/06_to_yolo.py 생성. 이미지 복제 없음 - 목록 파일로 가리킨다.\n"
            "path: %s\ntrain: train_box.txt\nval: %s\nnames:\n%s\n" % (root, val_list, names))
        return p
    made = [yml("seg.yaml", "val_seg.txt"),          # 학습·모델 선택. 사람 마스크만
            yml("test_seg.yaml", "test_seg.txt"),    # mask AP
            yml("test_box.yaml", "test_box.txt"),    # box AP (전량)
            yml("control.yaml", "control_box.txt"),  # 대조군 - box AP 만 (D14)
            yml("crossdomain.yaml", "crossdomain_seg.txt"),
            yml("test_ref.yaml", "test_ref.txt"),        # 참고치 - SAM 채움본 정답
            yml("control_ref.yaml", "control_ref.txt")]  # 참고치 (D14)

    log("\n=== 목록 ===")
    for k, (p, n) in sorted(lists.items()):
        log("  %-20s %5d" % (k, n))
    log("\n=== yaml ===")
    for m in made:
        log("  %s" % m)
    if stat["사진 없음"]:
        log("\n주의: 사진 없음 %d건" % stat["사진 없음"])
    for k in sorted(stat):
        if k.endswith("박스 대체"):
            log("박스로 대체한 인스턴스 - %s %d" % (k.split(":")[0], stat[k]))


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    main()
