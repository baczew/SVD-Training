from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os

#Loading image and setting parameters

plt.rcParams['figure.figsize'] = [16, 8]

image = imread('swiss.jpg')
print(image.shape)
X = np.mean(image, -1)
print(X.shape)
img = plt.imshow(X)

plt.set_cmap("gray")
plt.axis('off')


#Plain decomposition
U, S, V_T = np.linalg.svd(X, full_matrices = False) #full_mat argument indicates 'economy SVD'
S = np.diag(S)


#Approxiamtion function

def approximate_matrix(m, r):

    m_U, m_S, mV_T = np.linalg.svd(m, full_matrices = False)
    m_S = np.diag(m_S)
    m_approximation = m_U[:, :r] @ m_S[:r, :r] @ mV_T[:r, :]
    return m_approximation


#Trying different approximations

for r in [50, 200]:
    X_approximated = approximate_matrix(X, r)
    plt.imshow(X_approximated, cmap="gray")
    plt.axis("off")
    plt.title(f"Based on r={r}")
    plt.show()


#Plotting S values

plt.figure(10)
plt.title("Classic")
plt.plot(np.diag(S))

plt.figure(11)
plt.title("Log")
plt.semilogy(np.diag(S))

plt.figure(12)
plt.title("Cumulative")
plt.semilogy(np.cumsum(np.diag(S))/sum(np.diag(S)))

plt.show()







