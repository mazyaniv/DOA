import numpy as np
import math

def observ(SNR, snap, A):
    M = A.shape[0]
    D = A.shape[1]
    real_s = np.random.normal(0, 1 / math.sqrt(2), (D, snap))
    im_s = np.random.normal(0, 1 / math.sqrt(2), (D, snap))
    s = real_s + 1j * im_s
    s_samp = s.reshape(D, snap)

    real_n = np.random.normal(0, (10 ** (-SNR / 20)) / math.sqrt(2), (M, snap))
    im_n = np.random.normal(0, (10 ** (-SNR / 20)) / math.sqrt(2), (M, snap))
    n = real_n + 1j * im_n
    n_samp = n.reshape(M, snap)
    x_a_samp = (A@s_samp) + n_samp
    return x_a_samp

def quantize(A, P, thresh_real=0, thresh_im=0):
    mask = np.zeros(np.shape(A), dtype=complex)
    mask[:P, :] = (1 / math.sqrt(2)) * (
            np.sign(A[:P, :].real - (thresh_real)) + (1j * (np.sign(A[:P, :].imag - ((thresh_im))))))
    mask[P:, :] = A[P:, :]
    return mask

def covariance(v1, v2):
    normv1 = np.mean(v1, 1)
    normv2 = np.mean(v2, 1)
    v = v1 - normv1.reshape(np.shape(v1)[0], 1)
    u = v2 - normv2.reshape(np.shape(v2)[0], 1)
    result = [v[:, i].reshape(np.shape(v)[0], 1) @ u[:, i].conjugate().transpose().reshape(1, np.shape(u)[0]) for i in
              range(np.shape(v)[1])]
    return np.sum(result, 0) / (np.shape(v)[1] - 1)