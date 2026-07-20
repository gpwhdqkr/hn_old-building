import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 한글 폰트 설정 (사용 환경에 맞게 'Malgun Gothic', 'NanumGothic' 등으로 변경 가능)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 설정
periods = ['10년 미만', '10년~20년 미만', '20년~30년 미만', '30년 이상']
data_counts = {
    'A등급': [13463, 999, 742, 465],
    'B등급': [20890, 24689, 29366, 16422],
    'C등급': [102, 257, 697, 4686],
    'D등급': [0, 0, 6, 322],
    'E등급': [0, 0, 1, 50],
    '미지정': [166, 390, 191, 940],
}

df_counts = pd.DataFrame(data_counts, index=periods)

# 사용연수별 비율(%) 계산
df_percent = df_counts.div(df_counts.sum(axis=1), axis=0) * 100

# 그래프 색상 및 범주 설정
colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#7f7f7f']
ratings = df_percent.columns

# 100% 누적 막대그래프 생성
fig, ax = plt.subplots(figsize=(12, 6))
bottom = np.zeros(len(periods))

for i, col in enumerate(ratings):
    values = df_percent[col]
    bars = ax.bar(
        periods, values, bottom=bottom, label=col, color=colors[i], width=0.55
    )

    # 텍스트 오버랩 방지를 위해 3.5% 이상인 구간만 수치 표시
    for j, (val, b) in enumerate(zip(values, bottom)):
        if val >= 3.5:
            ax.text(
                j,
                b + val / 2,
                f'{val:.1f}%',
                ha='center',
                va='center',
                color='white' if i in [1, 3, 4] else 'black',
                fontweight='bold',
                fontsize=9,
            )

    bottom += values

# 축 라벨 및 디자인 설정
ax.set_ylabel('비율 (%)', fontsize=12, fontweight='bold')
ax.set_title(
    '사용연수별 건축물 안전등급 비율 (%)',
    fontsize=15,
    fontweight='bold',
    pad=15,
)
ax.set_ylim(0, 100)
ax.legend(
    title='안전등급',
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    frameon=True,
)
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('building_safety_percentage.png', dpi=300)
plt.show()