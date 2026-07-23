"""GPU 연결 확인용 초소형 학습 테스트 (약 1초면 끝남).

RunPod 팟(원격 GPU 서버) 안에서 실행하는 스크립트:
  1. runpod_test.py status <POD_ID> 로 Jupyter 주소를 확인해서 접속
  2. 이 파일을 업로드하거나 터미널에서 붙여넣기
  3. python gpu_test.py

CUDA가 잡히는지, GPU에서 실제로 학습(forward/backward/step)이 도는지 확인한다.
"""

import time

import torch
import torch.nn as nn


def main():
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 버전: {torch.version.cuda}")
    else:
        device = "cpu"
        print("경고: GPU가 잡히지 않아 CPU로 실행합니다. (팟에서 실행 중인지 확인하세요)")

    # 아주 작은 MLP + 랜덤 데이터로 1초 동안만 학습
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    x = torch.randn(256, 64, device=device)
    y = torch.randn(256, 1, device=device)

    start = time.time()
    steps = 0
    loss = None
    while time.time() - start < 1.0:  # 1초 시간제한
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        steps += 1

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"\n학습 완료: {device}에서 {elapsed:.2f}초 동안 {steps} step 수행")
    print(f"최종 loss: {loss.item():.4f}")
    if device == "cuda":
        print(f"GPU 메모리 사용: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
        print("\n=> GPU 연결 및 학습 정상 동작 확인!")


if __name__ == "__main__":
    main()
