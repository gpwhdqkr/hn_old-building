"""RunPod 네트워크 볼륨(스토리지)에 접속해서 GPU Pod을 돌리는 템플릿 (공식 runpod SDK).

여기서 "스토리지 접속"이란: 파이썬 SDK로 계정에 붙어 네트워크 볼륨(팀 공용 저장소)을
확인하고, 그 볼륨을 붙인 GPU Pod을 만드는 것까지다. 볼륨 안의 파일을 로컬에서 직접
열어보는 게 아니라, Pod에 마운트해서 Pod 안에서 파일 작업을 하게 된다.

전체 흐름:
  1) 파이썬 SDK로 계정 접속 -> 네트워크 볼륨(스토리지) 목록 확인   (volumes)
  2) GPU를 골라 그 볼륨을 붙인 Pod 생성                          (create)
     -> 볼륨이 /workspace 에 마운트됨
  3) Pod 안에서 구글드라이브 데이터를 /workspace 로 받아 학습     (status가 방법 안내)
     -> 데이터는 볼륨에 남으므로, 다음에 새 Pod를 만들어도 다시 안 받아도 됨
  4) 끝나면 Pod 정리 (과금 종료, 볼륨 데이터는 유지)             (stop / terminate)

사용 전 준비:
  1. https://www.runpod.io/console/user/settings 에서 API 키 발급 (팀 계정이면 팀장 키)
  2. 이 폴더의 .env 파일에  RUNPOD_API_KEY=본인키  한 줄 작성 (.env.example 참고)
     또는 PowerShell에서  $env:RUNPOD_API_KEY="본인키"
  3. pip install runpod
  ※ 네트워크 볼륨은 RunPod 콘솔 Storage 메뉴에서 미리 만들어 두어야 함.

사용법:
  python runpod_storage_test.py volumes                                            # 1) 내 볼륨 목록
  python runpod_storage_test.py gpus                                               # 고를 수 있는 GPU id
  python runpod_storage_test.py create --volume-id <VOL_ID> --gpu "NVIDIA GeForce RTX 3090"
  python runpod_storage_test.py status <POD_ID>                                    # 상태 + 접속 + 데이터받는법
  python runpod_storage_test.py stop <POD_ID>                                      # 정지 (GPU 반납)
  python runpod_storage_test.py terminate <POD_ID>                                 # 완전 삭제 (볼륨은 유지)
"""

import argparse
import os
import sys
from pathlib import Path

import runpod
from runpod.error import QueryError

# 기본값 (create 명령의 옵션으로 덮어쓸 수 있음)
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_GPU = "NVIDIA GeForce RTX 3090"
DEFAULT_DISK_GB = 20          # 컨테이너 디스크 (네트워크 볼륨과는 별개, 이미지/pip 설치용)
DEFAULT_MOUNT = "/workspace"  # 네트워크 볼륨이 마운트될 경로
DEFAULT_JUPYTER_PW = "runpod-test"  # Jupyter 접속 비밀번호 (Pod 생성 시 환경변수로 전달됨)


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

def cmd_volumes(args):
    """1) 계정에 접속해서 네트워크 볼륨(스토리지) 목록을 보여준다."""
    volumes = runpod.get_user().get("networkVolumes") or []
    if not volumes:
        print("네트워크 볼륨이 없습니다.")
        print("RunPod 콘솔 -> Storage -> New Network Volume 에서 먼저 볼륨을 만들어 주세요.")
        print("(볼륨을 붙이려면 볼륨과 '같은 데이터센터'에 GPU 재고가 있어야 합니다.)")
        return

    print(f"{'VOLUME_ID (--volume-id 로 사용)':30} {'이름':16} {'크기':>6}  데이터센터")
    print("-" * 74)
    for v in volumes:
        size = f"{v['size']}GB"
        print(f"{v['id']:30} {v['name']:16} {size:>6}  {v['dataCenterId']}")
    print('\nPod 생성:  python runpod_storage_test.py create --volume-id <VOLUME_ID> --gpu "<GPU id>"')


def cmd_gpus(args):
    """고를 수 있는 GPU id 목록 (create --gpu 에 그대로 넣는 값)."""
    gpus = runpod.get_gpus()
    print(f"고를 수 있는 GPU {len(gpus)}종 (아래 id를 create --gpu 값으로 사용):\n")
    print(f"{'GPU id':45} {'VRAM':>6}")
    print("-" * 55)
    for g in gpus:
        print(f"{g['id']:45} {g['memoryInGb']:>4}GB")
    print("\n※ 시간당 가격까지 보려면:  python runpod_test.py gpus")
    print("※ 볼륨이 있는 데이터센터에 그 GPU 재고가 없으면 create가 실패할 수 있어요. 그러면 다른 GPU로 재시도.")


def cmd_create(args):
    """2) 지정한 네트워크 볼륨을 붙여서 GPU Pod을 생성한다."""
    # 볼륨이 실제로 있는지 / 어느 데이터센터인지 먼저 확인 (친절한 에러를 위해)
    volumes = runpod.get_user().get("networkVolumes") or []
    volume = next((v for v in volumes if v["id"] == args.volume_id), None)
    if volume is None:
        have = ", ".join(v["id"] for v in volumes) or "(볼륨 없음)"
        sys.exit(
            f"볼륨을 찾을 수 없습니다: {args.volume_id}\n"
            f"  내 볼륨: {have}\n"
            f"  목록 확인:  python runpod_storage_test.py volumes"
        )

    print("Pod 생성 중...")
    print(f"  GPU:     {args.gpu}")
    print(f"  볼륨:    {volume['name']} ({volume['id']}, {volume['size']}GB) @ {volume['dataCenterId']}")
    print(f"  마운트:  {args.mount}  <- 이 경로에 볼륨이 붙습니다 (데이터는 여기에 저장하면 영구 보존)")
    print(f"  이미지:  {args.image}")
    print("※ 네트워크 볼륨은 Secure Cloud 전용이라 cloud_type=SECURE 로 만듭니다.")
    print("※ 생성되는 순간부터 과금 시작! 테스트가 끝나면 반드시 terminate 하세요.\n")

    try:
        pod = runpod.create_pod(
            name=args.name,
            image_name=args.image,
            gpu_type_id=args.gpu,
            cloud_type="SECURE",               # 네트워크 볼륨은 Secure Cloud에서만 붙일 수 있음
            gpu_count=1,
            network_volume_id=args.volume_id,  # 데이터센터는 SDK가 볼륨을 보고 자동으로 맞춰줌
            volume_mount_path=args.mount,      # 볼륨이 마운트될 경로 (기본 /workspace)
            container_disk_in_gb=args.disk,
            min_vcpu_count=2,
            min_memory_in_gb=8,
            ports="8888/http,22/tcp",          # Jupyter(8888) + SSH(22)
            env={"JUPYTER_PASSWORD": DEFAULT_JUPYTER_PW},
        )
    except QueryError as err:
        sys.exit(
            f"Pod 생성 실패: {err}\n"
            f"  '{volume['dataCenterId']}' 데이터센터에 '{args.gpu}' 재고가 없을 수 있습니다.\n"
            f"  다른 GPU로 다시 시도해 보세요 (id 목록: python runpod_storage_test.py gpus)."
        )

    print(f"Pod 생성 완료! POD_ID = {pod['id']}")
    print("\n다음 단계:")
    print(f"  접속정보/데이터 받는 법:  python runpod_storage_test.py status {pod['id']}")
    print(f"  정리(과금 종료):          python runpod_storage_test.py terminate {pod['id']}")


def cmd_status(args):
    """Pod 상태 + 접속 정보 + (3단계) 구글드라이브 데이터 받는 법 안내."""
    pod = runpod.get_pod(args.pod_id)
    if pod is None:
        sys.exit(f"Pod을 찾을 수 없습니다: {args.pod_id}")

    print(f"Pod: {pod['name']} ({pod['id']})")
    print(f"상태: {pod['desiredStatus']}, 요금: ${pod['costPerHr']}/hr")
    machine = pod.get("machine") or {}
    if machine.get("gpuDisplayName"):
        print(f"GPU: {machine['gpuDisplayName']}")

    runtime = pod.get("runtime")
    if not runtime:
        print("\n아직 부팅 중입니다. 1~2분 뒤 다시 status를 실행하세요.")
        return

    print("\n[접속]")
    print(f"  Jupyter Lab:  https://{pod['id']}-8888.proxy.runpod.net   (비밀번호: {DEFAULT_JUPYTER_PW})")
    if machine.get("podHostId"):
        print(f"  SSH:          ssh {machine['podHostId']}@ssh.runpod.io -i <SSH개인키경로>")
        print("                (RunPod 콘솔 Settings에 SSH 공개키를 등록해야 함)")

    print("\n[3) 볼륨에 데이터 넣기]  <- Jupyter 터미널이나 SSH로 Pod에 들어가서 실행")
    print(f"  pip install gdown")
    print(f"  cd {DEFAULT_MOUNT}                            # 볼륨이 마운트된 곳 (데이터는 여기에 저장)")
    print(f"  gdown <파일ID>                                # 파일 하나")
    print(f"  gdown --folder <폴더ID>                       # 폴더째")
    print(f"  # ID는 공유링크에서: 파일은 /d/ 와 /view 사이, 폴더는 folders/ 뒤 문자열")
    print(f"  # 링크 전체 말고 ID만 넣으세요. (팟의 gdown 버전에 따라 링크/--fuzzy 방식은 안 될 수 있음)")
    print(f"  # {DEFAULT_MOUNT}(볼륨)에 받아두면 Pod을 terminate 해도 데이터는 남습니다. (다음 Pod에서 재사용)")


def cmd_stop(args):
    """4) Pod 정지 - GPU 과금은 멈추고 볼륨 데이터는 유지."""
    pod = runpod.stop_pod(args.pod_id)
    print(f"Pod 정지됨: {pod['id']} (상태: {pod['desiredStatus']})")
    print("※ GPU 과금은 멈춥니다. 네트워크 볼륨의 데이터는 그대로 유지됩니다.")
    print(f"  완전히 끝내려면:  python runpod_storage_test.py terminate {pod['id']}")


def cmd_terminate(args):
    """4) Pod 완전 삭제 - Pod 과금 종료. 네트워크 볼륨과 데이터는 남는다."""
    runpod.terminate_pod(args.pod_id)
    print(f"Pod 삭제 완료: {args.pod_id} (Pod 과금 종료)")
    print("※ 네트워크 볼륨과 그 안의 데이터는 삭제되지 않고 남아있습니다.")
    print("  (볼륨 자체를 지우려면 RunPod 콘솔 -> Storage 에서 삭제하세요.)")


# ---------------------------------------------------------------- 진입점

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="RunPod 네트워크 볼륨(스토리지) + GPU Pod 도구 (공식 SDK)")
    parser.add_argument("--api-key", help="RunPod API 키 (기본: RUNPOD_API_KEY 환경변수 또는 .env)")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("volumes", help="내 네트워크 볼륨(스토리지) 목록")
    sub.add_parser("gpus", help="고를 수 있는 GPU id 목록")

    p_create = sub.add_parser("create", help="볼륨을 붙인 GPU Pod 생성 (과금 시작)")
    p_create.add_argument("--volume-id", required=True, help="붙일 네트워크 볼륨 id (volumes 명령으로 확인)")
    p_create.add_argument("--gpu", default=DEFAULT_GPU, help=f"GPU id (기본: {DEFAULT_GPU})")
    p_create.add_argument("--name", default="storage-train-pod", help="Pod 이름")
    p_create.add_argument("--image", default=DEFAULT_IMAGE, help="도커 이미지")
    p_create.add_argument("--disk", type=int, default=DEFAULT_DISK_GB, help="컨테이너 디스크 GB (볼륨과 별개)")
    p_create.add_argument("--mount", default=DEFAULT_MOUNT, help=f"볼륨 마운트 경로 (기본: {DEFAULT_MOUNT})")

    p_status = sub.add_parser("status", help="Pod 상태 + 접속정보 + 데이터 받는 법")
    p_status.add_argument("pod_id")

    p_stop = sub.add_parser("stop", help="Pod 정지 (GPU 반납, 볼륨 유지)")
    p_stop.add_argument("pod_id")

    p_term = sub.add_parser("terminate", help="Pod 완전 삭제 (볼륨 데이터는 유지)")
    p_term.add_argument("pod_id")

    args = parser.parse_args()

    # 명령 없이 실행하면(예: IDE에서 그냥 Run) 도움말을 보여주고 정상 종료
    if args.command is None:
        parser.print_help()
        return

    setup_api_key(args.api_key)

    commands = {
        "volumes": cmd_volumes,
        "gpus": cmd_gpus,
        "create": cmd_create,
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
