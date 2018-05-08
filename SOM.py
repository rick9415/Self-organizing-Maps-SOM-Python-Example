import matplotlib
import matplotlib.pyplot as plt
import numpy as np
A = np.array([[
    0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
    1, 0, 0, 0, 1, 1, 0, 0, 0, 1
]])
B = np.array([[
    1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1,
    1, 0, 0, 0, 1, 1, 1, 1, 1, 0
]])
C = np.array([[
    0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    1, 0, 0, 0, 1, 0, 1, 1, 1, 0
]])
D = np.array([[
    1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1,
    1, 0, 0, 1, 0, 1, 1, 1, 0, 0
]])
E = np.array([[
    1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 1, 1, 1, 1, 1
]])
F = np.array([[
    1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0
]])
G = np.array([[
    0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1,
    1, 0, 0, 0, 1, 0, 1, 1, 1, 0
]])
H = np.array([[
    1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1,
    1, 0, 0, 0, 1, 1, 0, 0, 0, 1
]])
I = np.array([[
    0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 1, 0, 0
]])
J = np.array([[
    0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
    1, 0, 0, 1, 0, 1, 1, 1, 1, 0
]])
K = np.array([[
    1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0,
    1, 0, 0, 1, 0, 1, 0, 0, 0, 1
]])
L = np.array([[
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 1, 1, 1, 1, 1
]])
X = np.concatenate(
    [A, B, C, D, E, F, G, H, I, J, K, L], axis=0)  # Data : (12,35)

k = 16
R0 = 1.5
lr0 = 0.1
t = 1000 / (np.log(R0))
t2 = 1000
Epochs = 1000  #10000  # 5000

w = np.random.randn(k, 35)  # (16,35)
N = np.zeros([len(X), k])  # (12,16)

for i in range(len(X)):  # 12
    for j in range(len(w)):  # 16
        N[i, j] = (np.linalg.norm(X[i] - w[j]))  # (12,16)

rq = np.zeros([len(X), 1, 2])  # (12,1,2)
for i in range(12):
    x = np.argmin(N[i, :])  # (12,1) 12 data 距 16 neuron最小距離 index (0~15)
    p = lambda x: (x // 4, x % 4)  # 轉成4x4的座標
    rq[i, :, :] = np.array([p(x)])  # 輸入與最短神經元的座標(12,1,2)

ri = np.zeros([k, 1, 2])
for i in range(k):
    q = lambda x: (x // 4, x % 4)
    ri[i] = np.array(q(i))  # ri 4*4 座標 (16,1,2)  [0,0],[0,1]...->[3,3]


def Draw_w(m, x=4, y=4):
    fig = plt.figure()
    m = np.reshape(m, (len(m), 7, 5))
    for num in range(len(m)):
        ax = fig.add_subplot(x, y, num + 1)  #num + 1: 1~16
        ax.matshow(1 - m[num, :, :], cmap=plt.cm.gray)
    plt.tight_layout()
    plt.show()


h = np.zeros([len(rq), len(ri)])  # (12,16)
for k in range(Epochs):
    lr = lr0 * np.exp(-k / t2)
    R = R0 * np.exp(-k / t)
    for q in range(len(rq)):  # X 12
        for j in range(len(ri)):  # 16
            d = ((np.linalg.norm(ri[j] - rq[q]))**2)
            h[q, j] = np.exp(-d / (2 * (R**2)))
    for q in range(len(X)):  # X 12
        for j in range(len(w)):  # 16
            w_grad = lr * h[q, j] * (X[q, :] - w[j, :])
            w[j, :] = w[j, :] + w_grad  #(16, 35)
h = np.round(h, 2)
print(h.shape)
print(h)
w = np.nan_to_num(w)
w = np.reshape(w, (len(w), 7, 5))
Draw_w(w)
