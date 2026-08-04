"""화질 축 증강 두 갈래. 의존성 없이 직접 구현한다.

왜 직접 만드나 — `albumentations` 가 이 환경에 설치돼 있지 않고, ultralytics 는
`except ImportError: pass` 로 조용히 건너뛴다. 지금까지의 12회 학습에는 화질 관련 증강이
**하나도 걸리지 않았다.** 지금 설치하면 기본 변환(Blur·MedianBlur·ToGray·CLAHE, 각 p=0.01)이
딸려 들어와 통제가 흐려지므로, 재려는 것만 직접 넣는다.

두 갈래인 이유는 실측이 기획의 전제와 어긋났기 때문이다(E8).

  (가) `Degrade` — **기획서 원안.** "줄인 사진은 실제보다 깨끗하니 노이즈·흐림·압축으로
       되돌린다." 물체 크기를 맞춰 재보니 축소본이 오히려 더 시끄러웠지만(대조군 3.21 vs
       축소 4.49), 원안을 재보지 않고 버리면 "전제가 틀렸다"와 "증강이 안 듣는다"가 갈리지 않는다.

  (나) `Flatten` — **실측이 가리킨 것.** 대조군과 학습 자료의 가장 큰 차이는 노이즈가 아니라
       **대비**였다(미헬슨 0.255 vs 0.588, 2.3배). 대비를 낮추고 옅은 안개를 씌워 그 격차를 줄인다.
       D12 는 "색·명암 크게 흔들기"를 껐는데, 그 근거는 금속 반사가 재질 단서라는 추론이었다.
       여기서 재는 것은 무작위 흔들기가 아니라 **측정된 격차만큼의 한 방향 이동**이다.

둘 다 기하 정보를 건드리지 않으므로 박스·마스크를 손댈 필요가 없다.
ASCII 주석.
"""
import random

import cv2
import numpy as np


class Degrade:
    """노이즈 · 흐림 · JPEG 압축. 기획서 06절 '화질 열화 증강' 원안."""

    def __init__(self, p=0.5, noise=(2.0, 8.0), blur=(0.3, 1.2), jpeg=(40, 85)):
        self.p, self.noise, self.blur, self.jpeg = p, noise, blur, jpeg

    def __call__(self, labels):
        if random.random() > self.p:
            return labels
        im = labels["img"]
        if random.random() < 0.7:                       # 흐림
            s = random.uniform(*self.blur)
            im = cv2.GaussianBlur(im, (3, 3), s)
        if random.random() < 0.7:                       # 센서 노이즈
            s = random.uniform(*self.noise)
            im = np.clip(im.astype(np.float32) +
                         np.random.normal(0, s, im.shape), 0, 255).astype(np.uint8)
        if random.random() < 0.7:                       # 압축 손상
            q = random.randint(*self.jpeg)
            ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if ok:
                im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        labels["img"] = im
        return labels


class Flatten:
    """대비를 낮추고 옅은 안개를 씌운다. E8 이 잰 격차를 되돌리는 방향.

    alpha 하한은 실측에서 왔다 - 대조군/학습자료 미헬슨 대비가 0.255/0.588 = 0.43.
    무작위로 밝기를 흔드는 것이 아니라 **한 방향(낮추는 쪽)으로만** 움직인다.
    """

    def __init__(self, p=0.5, alpha=(0.45, 1.0), haze=(0.0, 0.25), bright=(-12, 12)):
        self.p, self.alpha, self.haze, self.bright = p, alpha, haze, bright

    def __call__(self, labels):
        if random.random() > self.p:
            return labels
        im = labels["img"].astype(np.float32)
        a = random.uniform(*self.alpha)
        m = float(im.mean())
        im = m + (im - m) * a                            # 대비 축소
        h = random.uniform(*self.haze)
        if h > 0:                                        # 옅은 안개(밝은 쪽으로 blend)
            im = im * (1 - h) + 200.0 * h
        im = im + random.uniform(*self.bright)
        labels["img"] = np.clip(im, 0, 255).astype(np.uint8)
        return labels


def attach(dataset, degrade=False, flatten=False):
    """데이터셋 변환 사슬의 **Format 앞**에 끼워 넣는다.

    Format 이 이미지를 텐서로 바꾸므로 그 뒤에 넣으면 안 된다.
    """
    tf = getattr(dataset, "transforms", None)
    lst = getattr(tf, "transforms", None)
    if lst is None:
        return False
    idx = len(lst)
    for i, t in enumerate(lst):
        if type(t).__name__ == "Format":
            idx = i
            break
    add = []
    if degrade:
        add.append(Degrade())
    if flatten:
        add.append(Flatten())
    for j, t in enumerate(add):
        lst.insert(idx + j, t)
    return bool(add)
