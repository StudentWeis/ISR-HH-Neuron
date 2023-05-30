import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# HH 模型参数
C_m = 1.0
g_Na = 120.0
g_K = 36.0
g_L = 0.3
E_Na = 50.0
E_K = -77.0
E_L = -54.387
I_app = 6.21

# 时间步长和时间范围
dt = 0.01
t_end = 500
time = np.arange(0, t_end, dt)

# 初始条件
V = -65.0
m = 0.0529
h = 0.5961
n = 0.3177

# 维纳噪声参数
D = 15
color_filter = 9.0

def colored_noise(t_end, dt, D, color_filter):
    n = int(t_end / dt)
    f = np.fft.fftfreq(n, d=dt)
    psd = 2 * D * color_filter / (1 + (color_filter * f) ** 2)
    noise_fft = np.random.normal(size=n) * np.sqrt(psd)
    noise = np.fft.ifft(noise_fft).real
    return noise

def alpha_m(V):
    return 0.1 * (V + 40.0) / (1.0 - np.exp(-0.1 * (V + 40.0)))

def beta_m(V):
    return 4.0 * np.exp(-0.0556 * (V + 65.0))

def alpha_h(V):
    return 0.07 * np.exp(-0.05 * (V + 65.0))

def beta_h(V):
    return 1.0 / (1.0 + np.exp(-0.1 * (V + 35.0)))

def alpha_n(V):
    return 0.01 * (V + 55.0) / (1.0 - np.exp(-0.1 * (V + 55.0)))

def beta_n(V):
    return 0.125 * np.exp(-0.0125 * (V + 65.0))

plt.figure(figsize=(16, 6))
plt.subplot(121)
plt.title('Membrane Potential')
plt.xlabel('Time (ms)')
plt.ylabel('V (mV)')
plt.ylim(-80, 40)
plt.subplot(122)
plt.title('Power Spectral Density')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')

# 模拟三种不同的维纳噪声幅值
for c in [0.01, 0.05, 0.1]:
    noise = colored_noise(t_end, dt, D=c, color_filter=color_filter)
    V_list = []
    for t in time:
        # 计算各个离子通道的电流
        I_Na = g_Na * m ** 3 * h * (V - E_Na)
        I_K = g_K * n ** 4 * (V - E_K)
        I_L = g_L * (V - E_L)
        
        # 更新各个门控变量
        m += dt * (alpha_m(V) * (1 - m) - beta_m(V) * m)
        h += dt * (alpha_h(V) * (1 - h) - beta_h(V) * h)
        n += dt * (alpha_n(V) * (1 - n) - beta_n(V) * n)
        
        # 更新膜电位
        V += dt / C_m * (I_app + noise[int(t / dt)] - I_Na - I_K - I_L)
        V_list.append(V)
    
    # 绘制膜电位和功率谱密度图像
    plt.subplot(121)
    plt.plot(time, V_list)
    freq, Pxx_den = signal.welch(V_list, fs=1/dt, nperseg=1024)
    plt.subplot(122)
    plt.semilogy(freq, Pxx_den)

# 显示图像
plt.savefig('ISR.png')
plt.show()
