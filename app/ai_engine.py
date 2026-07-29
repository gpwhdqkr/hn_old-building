import os
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import convnext_tiny
import torchvision.transforms as transforms
from PIL import Image

# 🔒 [윈도우 콘솔 방어] cp949 콘솔에서 이모지 로그가 UnicodeEncodeError로
# 서버 기동을 죽이는 것을 차단 (표현 불가 문자는 ?로 대체, 리눅스에선 무해)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

# =========================================================================
# v3 서빙 규격 — src/v3 학습 파이프라인(preprocess_v3.py)과 반드시 동일해야 함
# =========================================================================
INPUT_SIZE = 448          # 모델 입력 크기 (CenterCrop) — 224가 아님에 주의
EVAL_RESIZE_SHORT = 512   # 평가 전처리 Resize 짧은 변

# 운영 임계값: tune_threshold_binclf_v3(v3cta 런)가 validation에서 선택한 값.
# 안전 검사는 놓침(불량→우수)이 치명적이므로 argmax 대신 "불량 recall >= 0.85
# 보장 지점"의 임계값을 쓴다. 재튜닝 시 환경변수 MODEL_THRESHOLD로 교체 가능.
OPERATING_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.50"))


class LayerCAM:
    """여러 층의 LayerCAM 맵을 뽑아 정규화-평균으로 융합한다.

    src/v3/layercam_binclf_v3.py에서 위치 채점(pointing game)에 검증한 구현을
    그대로 이식 — pytorch_grad_cam 외부 라이브러리 의존이 없다.
    얕은 층이 섞여 Grad-CAM보다 세밀해서 가는 균열 표시에 유리하다.
    """

    def __init__(self, model, modules):
        self.model = model
        self.activations = {}
        self.gradients = {}
        self.handles = []
        for index, module in enumerate(modules):
            self.handles.append(module.register_forward_hook(self._save_activation(index)))
            self.handles.append(module.register_full_backward_hook(self._save_gradient(index)))

    def _save_activation(self, index):
        def hook(module, inputs, output):
            self.activations[index] = output.detach()
        return hook

    def _save_gradient(self, index):
        def hook(module, grad_inputs, grad_outputs):
            self.gradients[index] = grad_outputs[0].detach()
        return hook

    def compute(self, input_tensor, class_index):
        """입력 1장(batch=1)에 대한 융합 히트맵 [INPUT_SIZE, INPUT_SIZE] (0~1) 반환."""
        self.activations.clear()
        self.gradients.clear()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)
        logits[0, class_index].backward()

        fused = None
        n_maps = 0
        for index in self.activations:
            activation = self.activations[index]          # [1, C, h, w]
            gradient = self.gradients.get(index)
            if gradient is None:
                continue
            cam = torch.relu((torch.relu(gradient) * activation).sum(dim=1, keepdim=True))
            cam = F.interpolate(
                cam, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False
            )[0, 0]
            value_range = cam.max() - cam.min()
            if value_range > 0:
                cam = (cam - cam.min()) / value_range
                fused = cam if fused is None else fused + cam
                n_maps += 1

        if fused is None or n_maps == 0:
            fused = torch.zeros(INPUT_SIZE, INPUT_SIZE)
        else:
            fused = fused / n_maps
        return fused.cpu().numpy()

    def clear_buffers(self):
        """요청 간 활성값/그래디언트 잔류로 메모리가 쌓이지 않게 비운다."""
        self.activations.clear()
        self.gradients.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


class ApartmentClassifier:
    def __init__(self, model_path, device):
        self.device = device
        # v3 이원화: 0=우수 / 1=불량 (기존 3클래스의 "보통"은 불량에 병합됨)
        self.class_names = {0: "우수", 1: "불량"}
        self.threshold = OPERATING_THRESHOLD

        # ❶ 모델 구조: src/v3/model_factory_v3.create_model("convnext_tiny", 2)와
        # 100% 동일해야 state_dict가 로드된다 — classifier[2](마지막 Linear)만 교체.
        # (classifier 전체를 갈아끼우면 LayerNorm2d/Flatten 키가 사라져 로드 실패)
        self.model = convnext_tiny()
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, 2)

        # ❷ [안전 보장 가드] 가중치 로드 에러 원천 차단
        if model_path.exists():
            try:
                # 🔒 [클라우드 필수] CPU 전용 서버에서도 로드되도록 map_location 보장
                # v3 체크포인트는 weights_only=True 로드 가능하게 저장돼 있음
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    acc = checkpoint.get("validation_accuracy", 0)
                    macro_f1 = checkpoint.get("validation_macro_f1", 0)
                    print(f"🎉 [AI 엔진] v3 체크포인트를 불러왔습니다! "
                          f"(val 정확도: {acc:.2%} / macro F1: {macro_f1:.4f})")
                else:
                    self.model.load_state_dict(checkpoint)
                    print("🎉 [AI 엔진] 일반 state_dict 형태의 가중치를 불러왔습니다!")
            except Exception as e:
                print(f"❌ [AI 엔진] 가중치 파일 로드 중 실패(구조 불일치): {e}")
        else:
            print(f"❌ [AI 엔진] 에러: {model_path.resolve()} 경로에서 모델 파일을 찾지 못했습니다!")

        self.model = self.model.to(self.device)
        self.model.eval()

        # ❸ [메모리 방어] LayerCAM 엔진을 생성자에서 단 한 번만 생성해 재사용.
        # 타깃: features 후반 3개 블록 (layercam_binclf_v3.py의 convnext 관례와 동일)
        self.cam = LayerCAM(self.model, [
            self.model.features[-3], self.model.features[-2], self.model.features[-1]
        ])

        # ❹ 전처리: v3 평가 경로와 동일 배율 (짧은 변 512 리사이즈, 종횡비 유지).
        # Resize(512)는 정수 인자 = 짧은 변 기준 — Resize((512,512)) 아님!
        # v3cta는 CLAHE 미사용 런이므로 서빙에도 CLAHE를 넣지 않는다.
        #
        # [5-crop TTA] 서빙 판정은 중앙 1곳이 아닌 네 모서리+중앙 5개 크롭의
        # 최대 불량 확률로 한다. 중앙 크롭만 쓰면 세로/가로로 긴 사진의 가장자리
        # 결함이 시야 밖으로 잘려 미탐이 되기 때문(실내 벽 가장자리 균열 사례).
        # 각 크롭은 학습(RandomResizedCrop)·평가와 동일 배율의 448 패치이므로
        # 모델 입력 분포는 그대로다 — 재학습 불필요, 출력 집계만 바뀐다.
        self.resize = transforms.Resize(EVAL_RESIZE_SHORT)
        self.to_input = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def _five_crop_offsets(resized_width, resized_height):
        """리사이즈 좌표계의 5-crop 좌상단 오프셋 (네 모서리 + 중앙).

        짧은 변이 512라 두 변 모두 INPUT_SIZE(448) 이상임이 보장된다.
        정사각형에 가까운 이미지는 오프셋이 겹칠 수 있어 중복 제거한다.
        """
        span_x = resized_width - INPUT_SIZE
        span_y = resized_height - INPUT_SIZE
        offsets = [
            (0, 0), (span_x, 0), (0, span_y), (span_x, span_y),
            (span_x // 2, span_y // 2),  # 중앙은 반드시 마지막 (CAM 우선순위 규칙)
        ]
        return offsets

    def predict_and_get_cam(self, file_path):
        """이미지 1장을 5-crop TTA로 추론해 프론트 연동 5종 출력의 원천 데이터를 반환한다.

        반환 튜플:
          prediction         : 0=우수, 1=불량 (운영 임계값 비교 판정)
          result_status      : "우수" / "불량"
          defect_probability : 5개 크롭 중 최대 불량 확률 0~1 (우수 확률 = 1 - 이 값)
          grayscale_cam      : 최대 확률 크롭의 LayerCAM 히트맵 448×448 (0~1) — 우수면 None
          cam_peak_xy        : 히트맵 최대점 (x, y) — 448 크롭 좌표계, 우수면 None
          crop_offset_xy     : 판정 근거 크롭의 좌상단 (x, y) — 리사이즈(짧은 변 512)
                               좌표계. cv_processor의 원본 역매핑에 필요. 우수면 None
          inference_time_ms  : 분류 + LayerCAM 포함 순수 연산 시간 (정수 ms)
        """
        # [파일 입출력/전처리는 시간 측정 범위 밖]
        image = Image.open(file_path).convert('RGB')
        resized = self.resize(image)
        resized_width, resized_height = resized.size
        offsets = self._five_crop_offsets(resized_width, resized_height)

        # 5개 크롭을 배치 1회 forward로 처리 (크롭별 5회 호출보다 빠름)
        crop_batch = torch.stack([
            self.to_input(resized.crop((x, y, x + INPUT_SIZE, y + INPUT_SIZE)))
            for x, y in offsets
        ]).to(self.device)

        # =========================================================================
        # ⏱️ [추론 시간 측정 시작] — GPU 환경(CUDA)일 때만 synchronize 가드
        # =========================================================================
        is_cuda = (self.device.type == 'cuda')
        if is_cuda:
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        # 1. 이원화 분류: 크롭별 불량 확률 → 최대값 판정 (메모리 절약 inference_mode)
        with torch.inference_mode():
            outputs = self.model(crop_batch)
            crop_probs = torch.softmax(outputs, dim=1)[:, 1]
            best_index = int(torch.argmax(crop_probs).item())
            defect_probability = crop_probs[best_index].item()

        # argmax가 아닌 운영 임계값 판정 (불량 recall 하한 보장 지점)
        prediction = 1 if defect_probability >= self.threshold else 0
        result_status = self.class_names[prediction]

        grayscale_cam = None
        cam_peak_xy = None
        crop_offset_xy = None

        # 2. 불량 판정일 때만 LayerCAM 역추적 (backward → enable_grad)
        #
        # [CAM 크롭 선정 규칙] 판정은 최대 확률이지만, CAM은 중앙 크롭이 스스로
        # 임계값을 넘으면 중앙 크롭에 건다. 확신 케이스는 여러 크롭이 확률 1.0
        # 동률이라 argmax가 임의의 모서리 크롭을 고르는데, 모서리 크롭은 결함이
        # 시야 가장자리에 걸려 박스 품질이 나빠진다(균열사진.jpg 회귀 사례).
        # 중앙이 임계값 미달일 때만(=중앙 단독으론 미탐이던 케이스) 모서리 크롭 사용.
        if prediction == 1:
            center_index = len(offsets) - 1  # _five_crop_offsets에서 중앙은 마지막
            if crop_probs[center_index].item() >= self.threshold:
                cam_index = center_index
            else:
                cam_index = best_index
            crop_offset_xy = offsets[cam_index]
            best_crop_tensor = crop_batch[cam_index:cam_index + 1].clone()
            with torch.enable_grad():
                grayscale_cam = self.cam.compute(best_crop_tensor, class_index=1)
            peak_row, peak_col = np.unravel_index(
                np.argmax(grayscale_cam), grayscale_cam.shape
            )
            cam_peak_xy = (int(peak_col), int(peak_row))

        if is_cuda:
            torch.cuda.synchronize()  # 🔒 GPU 연산 완료 대기

        # [추론 시간 측정 종료] 초 → 정수 밀리초(ms)
        inference_time_ms = int((time.perf_counter() - start_time) * 1000)
        # =========================================================================

        # 🔒 [클라우드 필수 - VRAM 방어 가드]
        # 다중 사용자 접근 시 LayerCAM 잔여 활성값/그래디언트가 메모리에 쌓여
        # OOM으로 서버가 터지는 현상을 차단한다.
        del crop_batch
        self.cam.clear_buffers()
        self.model.zero_grad(set_to_none=True)
        if is_cuda:
            torch.cuda.empty_cache()

        return (prediction, result_status, defect_probability,
                grayscale_cam, cam_peak_xy, crop_offset_xy, inference_time_ms)
