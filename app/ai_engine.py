import torch
from torch import nn
from torchvision.models import efficientnet_b0  # 🟢 EfficientNet-B0 전환 완료
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
import time  

class ApartmentClassifier:
    def __init__(self, model_path, device):
        self.device = device
        self.class_names = {0: "우수", 1: "보통", 2: "불량"}
        
        # ❶ 모델 구조 선언 
        self.model = efficientnet_b0()
        
        # ❷ 기존 가중치 저장 파일(.pth)의 파라미터 구조 이름과 100% 일치하도록 순차 레이어 변경
        input_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(input_features, 3)
        )
        
        # ❸ [안전 보장 가드] 가중치 로드 에러 원천 차단
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location=device)
                
                # 에폭/정확도가 포함된 딕셔너리 형태일 때와 일반 state_dict 형태일 때를 모두 대응합니다.
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    acc = checkpoint.get('validation_accuracy', 0)
                    print(f"🎉 [AI 엔진] 딕셔너리 형태의 가중치를 불러왔습니다! (정확도: {acc:.2%})")
                else:
                    self.model.load_state_dict(checkpoint)
                    print(f"🎉 [AI 엔진] 일반 state_dict 형태의 가중치를 불러왔습니다!")
            except Exception as e:
                print(f"❌ [AI 엔진] 가중치 파일 로드 중 실패(구조 불일치): {e}")
        else:
            print(f"❌ [AI 엔진] 에러: {model_path.resolve()} 경로에서 모델 파일을 찾지 못했습니다!")
            
        self.model = self.model.to(device)
        self.model.eval()
        
        # ❹ [메모리 방어] Grad-CAM 객체를 생성자에서 단 한 번만 생성하여 재사용합니다.
        # EfficientNet-B0의 최하단 합성곱 레이어인 features[-1]을 정확하게 조준합니다.
        self.cam = GradCAM(model=self.model, target_layers=[self.model.features[-1]])
        
        # 전처리 transform 설정
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_and_get_cam(self, file_path):
        """이미지 경로를 받아 예측 클래스 번호, 상태 텍스트, Grad-CAM 지도, 추론 속도(ms)를 반환"""
        start_time = time.time()  
        
        image = Image.open(file_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 1. AI 등급 추론 (메모리 절약을 위해 inference_mode 사용)
        with torch.inference_mode():
            outputs = self.model(image_tensor)
            prediction = outputs.argmax(dim=1).item()
            result_status = self.class_names.get(prediction, "알 수 없음")
            
        grayscale_cam = None
        
        # 2. 보통(1), 불량(2) 결함 상태일 때만 Grad-CAM 역추적 작동
        if prediction in[1, 2]:  
            # Grad-CAM은 그래디언트 역전파가 필요하므로 enable_grad()를 켭니다.
            with torch.enable_grad():
                targets = [ClassifierOutputTarget(prediction)]
                cam_input = image_tensor.clone()
                # 미리 생성해 둔 self.cam을 재사용하여 메모리 폭발을 막습니다.
                grayscale_cam = self.cam(input_tensor=cam_input, targets=targets)[0]
            
        # 소요 시간 연산 후 정수 ms로 반환
        inference_time = int((time.time() - start_time) * 1000)
        return prediction, result_status, grayscale_cam, inference_time