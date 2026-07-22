"""RunPod GPU 대여/관리 템플릿 (공식 runpod SDK 사용).

API 키로 RunPod에 연결해서 GPU 팟(Pod)을 검색/생성/조회/정지/삭제한다.

사용 전 준비:
  1. https://www.runpod.io/console/user/settings 에서 API 키 발급
  2. API 키 설정 - 둘 중 하나:
     - 이 폴더의 .env 파일에  RUNPOD_API_KEY=본인키  한 줄 작성 (.env.example 참고)
     - 또는 환경변수:  PowerShell에서  $env:RUNPOD_API_KEY="본인키"
  3. pip install runpod

사용법:
  python runpod_test.py gpus                                   # 대여 가능한 GPU 목록 + 시간당 가격
  python runpod_test.py create --gpu "NVIDIA GeForce RTX 3090" # GPU 팟 생성 (대여 시작 = 과금 시작!)
  python runpod_test.py list                                   # 내 팟 목록
  python runpod_test.py status <POD_ID>                        # 팟 상태 + 접속 정보(Jupyter/SSH)
  python runpod_test.py stop <POD_ID>                          # 팟 정지 (GPU 과금 중단, 디스크 요금은 유지)
  python runpod_test.py terminate <POD_ID>                     # 팟 완전 삭제 (모든 과금 종료, 데이터 삭제)
"""

import argparse
import os
import sys
from pathlib import Path

import runpod
from runpod.error import QueryError

# 팟 생성 시 기본값 (필요하면 create 명령의 옵션으로 덮어쓸 수 있음)
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_GPU = "NVIDIA GeForce RTX 3090"
DEFAULT_DISK_GB = 20      # 컨테이너 디스크
DEFAULT_VOLUME_GB = 20    # /workspace 영구 볼륨
DEFAULT_JUPYTER_PW = "runpod-test"  # Jupyter 접속 비밀번호 (팟 생성 시 환경변수로 전달됨)


def load_dotenv():
    """스크립트 옆의 .env 파일에서 KEY=VALUE를 읽어 환경변수로 등록한다."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def setup_api_key(cli_key=None):
    key = cli_key or os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit(
            "오류: RunPod API 키가 없습니다.\n"
            "  .env 파일에 RUNPOD_API_KEY=본인키 를 넣거나,\n"
            '  PowerShell에서 $env:RUNPOD_API_KEY="본인키" 를 실행하세요.\n'
            "  키 발급: https://www.runpod.io/console/user/settings"
        )
    runpod.api_key = key


# ---------------------------------------------------------------- 명령들

def cmd_gpus(args):
    gpus = runpod.get_gpus()
    print(f"GPU {len(gpus)}종 조회됨. 가격 조회 중... (GPU마다 API를 한 번씩 호출하므로 잠시 걸립니다)\n")
    print(f"{'GPU (id로 사용)':45} {'VRAM':>6} {'OnDemand$/hr':>13} {'Spot$/hr':>9}")
    print("-" * 80)
    for g in gpus:
        try:
            detail = runpod.get_gpu(g["id"])
            price = detail.get("lowestPrice") or {}
            ondemand = price.get("uninterruptablePrice")
            spot = price.get("minimumBidPrice")
        except QueryError:
            ondemand = spot = None
        od = f"{ondemand:.2f}" if ondemand else "-"
        sp = f"{spot:.2f}" if spot else "-"
        print(f"{g['id']:45} {g['memoryInGb']:>4}GB {od:>13} {sp:>9}")
    print('\n팟 생성:  python runpod_test.py create --gpu "<GPU id>"')


def cmd_create(args):
    cloud = "COMMUNITY" if args.community else "SECURE"
    print(f"팟 생성 중... (GPU: {args.gpu}, cloud: {cloud}, image: {args.image})")
    print("※ 생성이 완료되는 순간부터 과금이 시작됩니다. 테스트가 끝나면 반드시 terminate 하세요!\n")

    try:
        pod = runpod.create_pod(
            name=args.name,
            image_name=args.image,
            gpu_type_id=args.gpu,
            cloud_type=cloud,
            gpu_count=1,
            container_disk_in_gb=args.disk,
            volume_in_gb=args.volume,
            volume_mount_path="/workspace",
            min_vcpu_count=2,
            min_memory_in_gb=8,
            ports="8888/http,22/tcp",
            env={"JUPYTER_PASSWORD": DEFAULT_JUPYTER_PW},
        )
    except QueryError as err:
        sys.exit(
            f"팟 생성 실패: {err}\n"
            "해당 GPU의 재고가 없을 수 있습니다. 다른 GPU나 --community 옵션을 시도해보세요."
        )

    print(f"팟 생성 완료! POD_ID = {pod['id']}")
    print(f"\n다음 단계:")
    print(f"  상태/접속정보 확인:  python runpod_test.py status {pod['id']}")
    print(f"  테스트 후 삭제:      python runpod_test.py terminate {pod['id']}")


def cmd_list(args):
    pods = runpod.get_pods()
    if not pods:
        print("실행 중인 팟이 없습니다.")
        return
    print(f"{'POD_ID':20} {'이름':20} {'상태':10} {'$/hr':>6}  GPU")
    print("-" * 80)
    for p in pods:
        gpu = (p.get("machine") or {}).get("gpuDisplayName", "-")
        print(f"{p['id']:20} {p['name']:20} {p['desiredStatus']:10} {p['costPerHr']:>6}  {gpu}")


def cmd_status(args):
    pod = runpod.get_pod(args.pod_id)
    if pod is None:
        sys.exit(f"팟을 찾을 수 없습니다: {args.pod_id}")

    print(f"팟: {pod['name']} ({pod['id']})")
    print(f"상태: {pod['desiredStatus']}, 요금: ${pod['costPerHr']}/hr")
    machine = pod.get("machine") or {}
    if machine.get("gpuDisplayName"):
        print(f"GPU: {machine['gpuDisplayName']}")

    runtime = pod.get("runtime")
    if not runtime:
        print("\n아직 부팅 중입니다. 잠시 후 다시 status를 실행하세요.")
        return

    if runtime.get("uptimeInSeconds") is not None:
        print(f"가동 시간: {runtime['uptimeInSeconds']}초")
    for g in runtime.get("gpus") or []:
        print(f"GPU 사용률: {g.get('gpuUtilPercent')}%, VRAM 사용률: {g.get('memoryUtilPercent')}%")

    print("\n[접속 방법]")
    print(f"  Jupyter Lab:  https://{pod['id']}-8888.proxy.runpod.net")
    print(f"                (이 스크립트의 create로 만든 팟이라면 비밀번호: {DEFAULT_JUPYTER_PW})")
    if machine.get("podHostId"):
        print(f"  SSH:          ssh {machine['podHostId']}@ssh.runpod.io -i <SSH개인키경로>")
        print("                (RunPod 콘솔 Settings에 SSH 공개키를 등록해야 함)")
    for port in runtime.get("ports") or []:
        if port.get("isIpPublic") and port.get("type") == "tcp":
            print(f"  직접 TCP:     {port['ip']}:{port['publicPort']} -> 내부 {port['privatePort']}")


def cmd_stop(args):
    pod = runpod.stop_pod(args.pod_id)
    print(f"팟 정지됨: {pod['id']} (상태: {pod['desiredStatus']})")
    print("※ GPU 과금은 멈추지만 디스크 보관 요금은 계속 나옵니다. 완전히 끝내려면 terminate 하세요.")


def cmd_terminate(args):
    runpod.terminate_pod(args.pod_id)
    print(f"팟 삭제 완료: {args.pod_id} (과금 종료, 팟 안의 데이터도 삭제됨)")


# ---------------------------------------------------------------- 진입점

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="RunPod GPU 대여/관리 도구 (공식 SDK)")
    parser.add_argument("--api-key", help="RunPod API 키 (기본: RUNPOD_API_KEY 환경변수 또는 .env)")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("gpus", help="대여 가능한 GPU 목록/가격")

    p_create = sub.add_parser("create", help="GPU 팟 생성 (과금 시작)")
    p_create.add_argument("--gpu", default=DEFAULT_GPU, help=f"GPU 종류 id (기본: {DEFAULT_GPU})")
    p_create.add_argument("--name", default="gpu-test-pod", help="팟 이름")
    p_create.add_argument("--image", default=DEFAULT_IMAGE, help="도커 이미지")
    p_create.add_argument("--disk", type=int, default=DEFAULT_DISK_GB, help="컨테이너 디스크 GB")
    p_create.add_argument("--volume", type=int, default=DEFAULT_VOLUME_GB, help="/workspace 볼륨 GB")
    p_create.add_argument("--community", action="store_true", help="Community Cloud 사용 (더 저렴)")

    sub.add_parser("list", help="내 팟 목록")

    p_status = sub.add_parser("status", help="팟 상태 + 접속 정보")
    p_status.add_argument("pod_id")

    p_stop = sub.add_parser("stop", help="팟 정지")
    p_stop.add_argument("pod_id")

    p_term = sub.add_parser("terminate", help="팟 완전 삭제 (과금 종료)")
    p_term.add_argument("pod_id")

    args = parser.parse_args()

    # 명령 없이 실행하면(예: IDE에서 그냥 Run) 도움말을 보여주고 정상 종료
    if args.command is None:
        parser.print_help()
        return

    setup_api_key(args.api_key)

    commands = {
        "gpus": cmd_gpus,
        "create": cmd_create,
        "list": cmd_list,
        "status": cmd_status,
        "stop": cmd_stop,
        "terminate": cmd_terminate,
    }
    try:
        commands[args.command](args)
    except QueryError as err:
        sys.exit(f"RunPod API 오류: {err}")


if __name__ == "__main__":
    main()
