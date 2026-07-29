import sys
try:
    from flask import Flask, render_template
except ImportError:
    print("\n[에러] flask 라이브러리가 설치되지 않았습니다!")
    print("터미널에 'pip install flask'를 입력해 설치하세요.\n")
    sys.exit(1)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('scan.html')

if __name__ == '__main__':
    try:
        print("\n--- 플라스크 서버 구동 시도 중 ---")
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"\n[서버 구동 실패 사유]: {e}\n")