import numpy as np
import matplotlib.pyplot as plt
import scienceplots # scienceplots 임포트

# --- 커널 함수 정의 ---
def epanechnikov(t):
  """Epanechnikov 커널 함수"""
  return np.where(np.abs(t) < 1, 0.75 * (1 - t**2), 0)

def tricube(t):
  """Tri-cube 커널 함수"""
  return np.where(np.abs(t) < 1, (70/81) * (1 - np.abs(t)**3)**3, 0)

def gaussian(t):
  """Gaussian 커널 함수"""
  return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * t**2)

# --- 데이터 생성 ---
t = np.linspace(-2, 2, 500)

# 각 커널 함수에 대해 y값 계산
y_epa = epanechnikov(t)
y_tri = tricube(t)
y_gauss = gaussian(t)

# --- 그래프 그리기 ---
plt.style.use(['science', 'nature'])

fig, ax = plt.subplots(figsize=(6,4))

ax.plot(t, y_epa, label='Epanechnikov')
ax.plot(t, y_tri, label='Tri-cube')
ax.plot(t, y_gauss, label='Gaussian')

ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$D(t)$')

ax.set_title('Comparison of Kernel Functions')

ax.legend()

ax.axhline(0, color='gray', linewidth=0.5)
ax.set_ylim(bottom=-0.05) # y축 아래 약간의 여백 추가

fig.savefig('kernels.png', dpi=600, bbox_inches='tight') # 그래프를 파일로 저장
