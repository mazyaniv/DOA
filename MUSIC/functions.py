import numpy as np
import math
import scipy.signal as ss

def get_key_by_value(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key

def generate_qpsk_symbols(K,D):
    random_symbols = np.random.randint(0, 4, (K,D))
    qpsk_constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
    qpsk_symbols = qpsk_constellation[random_symbols]
    return qpsk_symbols.T

def observ(SNR, snap, A):
    M = A.shape[0]
    D = A.shape[1]

    sample_rate = 1e6
    t = np.arange(snap)/sample_rate  #time vector
    f_tone = np.array([0.02e6, 0.08e6])
    s = np.exp(2j * np.pi * f_tone.reshape(2,1) * t)
    # real_s = np.random.normal(1, 1 / math.sqrt(2), (D, snap))
    # im_s = np.random.normal(1, 1 / math.sqrt(2), (D, snap))
    # s = real_s + 1j * im_s

    # s_samp = generate_qpsk_symbols(snap,D)#s.reshape(D, snap)
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

def angles_generate(pram):
    while True:
        range_array = np.arange(pram.teta_range[0], pram.teta_range[1], pram.Res)
        teta = np.random.choice(range_array[1:-1], size=pram.D, replace=False)
        if pram.D == 1 or abs(teta[0] - teta[1]) > pram.delta:
            break
    return np.sort(teta)[::-1]

def music(pram,R):
    eigvals, eigvecs = np.linalg.eig(R)
    sorted_indices = np.argsort(eigvals.real)[::-1]  # Sort eigenvalues in descending order
    eigvecs_sorted = eigvecs[:, sorted_indices]
    En = eigvecs_sorted[:, pram.D:]

    theta_range = np.radians(np.arange(pram.teta_range[0], pram.teta_range[1], pram.Res))  # Convert angles to radians
    music_spectrum = np.zeros(len(theta_range))
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
    return pred*pram.Res+pram.teta_range[0]

def root_music(pram,R):
    my_vec_coff = np.zeros((pram.M, 2 * pram.M - 1), dtype=complex)
    eigvals, eigvecs = np.linalg.eig(R)
    sorted_indices = np.argsort(eigvals.real)[::-1]  # Sort eigenvalues in descending order
    eigvecs_sorted = eigvecs[:, sorted_indices]
    En = eigvecs_sorted[:, pram.D:]
    matrix = En@En.conj().T
    for i in range(np.shape(matrix)[1]):
        vector = np.concatenate((matrix[:, i][::-1].reshape(1, -1), np.zeros((1, pram.M - 1))), axis=1)
        my_vec_coff[i, :] = np.roll(vector, i)

    cofficients = np.sum(my_vec_coff, 0)
    roots = np.poly1d(cofficients[::-1]).r
    sorted_roots = sorted(roots, key=lambda r: abs(abs(r) - 1))
    closest_roots = sorted_roots[0], sorted_roots[2]  # two closest roots
    pred = -np.degrees(np.arcsin(np.angle(closest_roots) / math.pi))[::-1]  # TODO why "-"?
    pred = np.sort(pred)[::-1]
    return pred

def esprit(pram,R):
    eigvals, eigvecs = np.linalg.eig(R)
    sorted_indices = np.argsort(eigvals.real)[::-1]  # Sort eigenvalues in descending order
    eigvecs_sorted = eigvecs[:, sorted_indices]
    Es = eigvecs_sorted[:, :pram.D]
    S1 = Es[1:,:]
    S2 = Es[:-1,:]
    P = np.linalg.inv(S1.conj().transpose()@S1)@S1.conj().transpose()@S2 #LS
    eigvals, eigvecs = np.linalg.eig(P)
    pred = np.degrees(np.arcsin(np.angle(eigvals) / math.pi))
    pred = np.sort(pred)[::-1]
    return pred




