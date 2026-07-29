import os
import socket
from flask import Flask, send_from_directory

app = Flask(__name__, static_folder='.')

@app.route('/')
def index():
    # templates 폴더 안에 있는 finally.html 파일을 반환합니다.
    return send_from_directory('templates', 'finally.html')

@app.route('/<path:path>')
def serve_static(path):
    # [수정 완료] 에러의 원인이던 BASE_DIR을 지우고 깔끔하게 'templates' 경로로 지정했습니다.
    return send_from_directory('templates', path)

def get_local_ip():
    """같은 와이파이 내 다른 PC가 접속할 수 있도록 내 컴퓨터의 로컬 IP를 찾습니다."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

if __name__ == '__main__':
    ip_address = get_local_ip()
    print("\n" + "="*50)
    print(" 🚀 AI 노후 건물 진단 시스템 서버가 시작되었습니다.")
    print(f" 🔗 현재 PC에서 접속: http://localhost:5000")
    print(f" 🌐 다른 PC/폰에서 접속 (동일 Wi-Fi): http://{ip_address}:5000")
    print("="*50 + "\n")
    
    # host='0.0.0.0' 설정을 통해 동일 와이파이 내 다른 기기의 접근을 허용합니다.
    app.run(host='0.0.0.0', port=5000, debug=True)