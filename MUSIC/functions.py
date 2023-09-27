import numpy as np
import math
import scipy.signal as ss

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
    # mask = np.zeros(np.shape(A), dtype=complex)
    # mask[P, :] = A[P, :]
    # mask[:P, :] = (1 / math.sqrt(2)) * (
    #         np.sign(A[:P, :].real - (thresh_real)) + (1j * (np.sign(A[:P, :].imag - ((thresh_im))))))
    # mask[P+1:, :] = (1 / math.sqrt(2)) * (
    #         np.sign(A[P+1:, :].real - (thresh_real)) + (1j * (np.sign(A[P+1:, :].imag - ((thresh_im))))))
    # return mask
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

def covariance_matrix(num_angles,theta_range,pram,R):
    eigvals, eigvecs = np.linalg.eig(R)
    sorted_indices = np.argsort(eigvals.real)[::-1]  # Sort eigenvalues in descending order
    eigvecs_sorted = eigvecs[:, sorted_indices]
    En = eigvecs_sorted[:, pram.D:]

    music_spectrum = np.zeros(num_angles)
    for idx, theta in enumerate(theta_range):
        steering_vector = np.exp(-1j * np.pi * np.arange(pram.M) * np.sin(theta))
        music_spectrum[idx] = 1 / np.linalg.norm(En.conj().T @ steering_vector)
    # plt.plot(np.degrees(theta_range), music_spectrum)
    # plt.title(f"N_a={pram.M-pram.N_q}, N_q={pram.N_q},theta={teta},method={method}")
    # plt.show()

    peaks, _ = ss.find_peaks(music_spectrum)
    peaks = list(peaks)
    peaks.sort(key=lambda x: music_spectrum[x])
    pred = np.array(peaks[-pram.D:])
    pred = np.sort(pred)[::-1]
    return pred
