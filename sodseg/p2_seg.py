"""P2(stride 4) 분할 헤드를 쓰기 위한 검증기 보정.

ultralytics 8.4.113 의 `SegmentationValidator` 는 "마스크 원형(proto)의 해상도가 항상
imgsz/4" 라는 전제를 **두 곳에 상수 4 로 박아뒀다.**

  1) postprocess : `imgsz = [4 * x for x in proto.shape[2:]]`
     -> 원형에서 이미지 크기를 되계산한다. P2 는 원형이 imgsz/2 라 이미지 크기를 2배로
        잘못 잡고, 그 값으로 박스를 원형 좌표에 맞추므로 마스크가 어긋난다.
  2) _prepare_batch : `s // 4`
     -> 정답 마스크를 `mask_ratio` 와 무관하게 항상 imgsz/4 로 맞춘다.

그래서 P2 를 붙이면 검증에서 정답(imgsz/4)과 예측(imgsz/2)의 화소 수가 4배 어긋나
`mat1 and mat2 shapes cannot be multiplied` 로 죽는다. 설치본의 p2 설정(`v8`, `26`)이
전부 탐지 전용인 것도 이 때문으로 보인다.

고치는 방법은 상수를 모델에서 계산하는 것뿐이다.
원형 배율 = 가장 촘촘한 stride / 2  (P3 헤드: 8/2=4, P2 헤드: 4/2=2)

ASCII 주석.
"""
import torch
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.segment import SegmentationTrainer, SegmentationValidator
from ultralytics.utils import ops


def proto_ratio_of(model, default=4):
    """imgsz / proto 해상도. 가장 촘촘한 stride 의 절반이다(P3:4, P2:2).

    최종 검증은 `AutoBackend` 로 다시 싣는데 그쪽 `stride` 는 **최대값 32 스칼라**라
    stride 목록을 볼 수 없다. 그래서 이 값은 첫 추정치일 뿐이고, 실제 배율은
    입력 크기 / proto 크기로 매 배치 다시 잡는다(`postprocess`).
    """
    for obj in (model, getattr(model, "model", None)):
        s = getattr(obj, "stride", None)
        if s is None:
            continue
        try:
            vals = [float(x) for x in s]
        except TypeError:
            continue
        if len(vals) > 1:
            return max(1, int(min(vals)) // 2)
    return default


class P2SegmentationValidator(SegmentationValidator):
    """상수 4 를 모델에서 계산한 배율로 바꾼 것 외에는 원본과 같다."""

    proto_ratio = 4

    def init_metrics(self, model):
        super().init_metrics(model)
        self.proto_ratio = proto_ratio_of(model)
        self._img_hw = None

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        self._img_hw = tuple(batch["img"].shape[2:])      # 레터박스 후 실제 입력 크기
        return batch

    def postprocess(self, preds):
        proto = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        if getattr(self, "_img_hw", None):
            # stride 를 못 믿는 경로(AutoBackend)가 있어 실측으로 배율을 다시 잡는다
            r = int(round(self._img_hw[0] / proto.shape[2]))
            if r >= 1:
                self.proto_ratio = r
        out = DetectionValidator.postprocess(self, preds[0])
        imgsz = [self.proto_ratio * x for x in proto.shape[2:]]   # 원본은 4 * x
        for i, pred in enumerate(out):
            coefficient = pred.pop("extra")
            pred["masks"] = self.process(proto[i], coefficient, pred["bboxes"], shape=imgsz)
        return out

    def _prepare_batch(self, si, batch):
        pb = DetectionValidator._prepare_batch(self, si, batch)
        nl = pb["cls"].shape[0]
        if self.args.overlap_mask:
            masks = batch["masks"][si]
            index = torch.arange(1, nl + 1, device=masks.device).view(nl, 1, 1)
            masks = (masks == index).float()
        else:
            masks = batch["masks"][batch["batch_idx"] == si]
        if nl:
            r = self.proto_ratio                                   # 원본은 4 고정
            mask_size = [s if self.process is ops.process_mask_native else s // r
                         for s in pb["imgsz"]]
            if list(masks.shape[1:]) != list(mask_size):
                masks = F.interpolate(masks[None], mask_size, mode="bilinear",
                                      align_corners=False)[0]
                masks = masks.gt_(0.5)
        pb["masks"] = masks
        return pb


class P2SegmentationTrainer(SegmentationTrainer):
    def get_validator(self):
        from copy import copy
        return P2SegmentationValidator(self.test_loader, save_dir=self.save_dir,
                                       args=copy(self.args), _callbacks=self.callbacks)


def make_trainer(base=None, degrade=False, flatten=False):
    """P2 검증기 보정 + 화질 증강 주입을 함께 거는 트레이너를 만든다.

    화질 증강은 **학습 데이터셋에만** 건다. 평가에 걸면 무엇을 재는지 알 수 없다.
    """
    from ultralytics.models.yolo.segment import SegmentationTrainer as _ST
    from sodseg.degrade import attach
    parent = base or P2SegmentationTrainer

    class _T(parent):
        def build_dataset(self, img_path, mode="train", batch=None):
            ds = super().build_dataset(img_path, mode, batch)
            if mode == "train" and (degrade or flatten):
                from ultralytics.utils import LOGGER
                ok = attach(ds, degrade=degrade, flatten=flatten)
                LOGGER.info("[화질축] 학습 변환 주입 %s (degrade=%s flatten=%s)"
                            % ("성공" if ok else "실패", degrade, flatten))
            return ds

    return _T
