# RunPod GPU 대여 테스트

RunPod API 키로 GPU를 대여해서 학습이 도는지 확인하는 최소 템플릿.

## 파일 구성

| 파일 | 역할 | 실행 위치 |
|---|---|---|
| `runpod_test.py` | GPU 검색/대여/상태확인/반납 | 내 PC (로컬) |
| `gpu_test.py` | 1초짜리 초소형 학습으로 GPU 동작 확인 | RunPod 팟 (원격 GPU) |

## 준비

```powershell
pip install runpod
```

1. https://www.runpod.io/console/user/settings → **API Keys** 에서 키 발급
2. `.env.example`을 `.env`로 복사하고 키 입력:
   ```
   RUNPOD_API_KEY=발급받은키
   ```
3. RunPod 계정에 크레딧 충전 (Billing 메뉴, 최소 $10)

## 사용 흐름

```powershell
# 1. 대여 가능한 GPU와 가격 확인
python runpod_test.py gpus

# 2. GPU 팟 생성 (이 순간부터 과금 시작!)
python runpod_test.py create --gpu "NVIDIA GeForce RTX 3090" --community

# 3. 부팅 완료 여부 + 접속 주소 확인 (1~2분 걸릴 수 있음)
python runpod_test.py status <POD_ID>

# 4. 출력된 Jupyter Lab 주소로 브라우저 접속 (비밀번호: runpod-test)
#    → gpu_test.py를 업로드하고 터미널에서 실행:
#    python gpu_test.py

# 5. 테스트 끝나면 반드시 삭제 (과금 종료)
python runpod_test.py terminate <POD_ID>
```

## GPU에서 이런 출력이 나오면 성공

```
PyTorch 버전: 2.4.0+cu124
CUDA 사용 가능: True
GPU: NVIDIA GeForce RTX 3090
...
학습 완료: cuda에서 1.00초 동안 XXXX step 수행
=> GPU 연결 및 학습 정상 동작 확인!
```

## 주의사항

- **팟이 살아있는 동안 계속 과금됩니다.** 테스트 후 `terminate`를 꼭 실행하세요.
  `python runpod_test.py list` 로 남아있는 팟이 없는지 확인하는 습관을 들이면 안전합니다.
- `stop`은 GPU 과금만 멈추고 디스크 보관 요금은 계속 나갑니다. 완전 종료는 `terminate`.
- `--community` 옵션은 Community Cloud(개인 제공 서버)라서 더 저렴합니다. 테스트 용도로 충분합니다.
- `.env` 파일(API 키)은 `.gitignore`에 넣어놨으니 절대 커밋되지 않게 유지하세요.
- 이 스크립트는 공식 `runpod` SDK를 사용합니다 (`pip install runpod`).
