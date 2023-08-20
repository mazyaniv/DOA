import numpy as np
import math
import scipy.signal as ss
from numpy import linalg as LA
from functions import covariance, quantize, observ
from classes import Matrix_class

def music_algorithm(pram,method=0):
    N_a = pram.M-pram.N_q
    theta_range = np.radians(np.arange(pram.teta_range[0], pram.teta_range[1], 1))  # Convert angles to radians
    num_angles = len(theta_range)
    rho = pram.D * (10 ** (-pram.SNR / 10) + 1)
    labels = np.zeros((pram.monte, pram.D))
    teta_vector = np.zeros((pram.monte, pram.D))
    #print(teta_vector.shape)
    for i in range(0, pram.monte): #Create labels
        if pram.D == 1:
            teta = np.random.randint(pram.teta_range[0], pram.teta_range[1], size=pram.D)
        else:
            while True:
                teta = np.random.randint(pram.teta_range[0], pram.teta_range[1], size=pram.D)
                if teta[0] != teta[1]:
                    break
            teta = np.sort(teta)[::-1]
        labels[i, :] = teta

    for i in range(pram.monte):
        A = Matrix_class(pram.M, labels[i, :]).matrix()
        my_vec = observ(pram.SNR, pram.snapshot, A)
        my_vec = quantize(my_vec, pram.N_q)
        R = covariance(my_vec, my_vec)
        if method == 1:  #quantized_sin
            R[:pram.N_q,:pram.N_q] = rho * (np.sin((math.pi / 2) * R.real)
                                            + 1j * np.sin((math.pi / 2) * R.imag)) #R_quantize
            R[pram.N_q:,:pram.N_q] = #R_mixed
            R[:pram.N_q,pram.N_q:] = #R_mixed
            R[pram.N_q:,pram.N_q:] = R[pram.N_q:,pram.N_q:] #R_analog
        elif method == 2:  #quantized_lin
            R = (rho * math.pi / 2) * (np.subtract(R, 1 - (2 / math.pi) * np.identity(pram.M)))

        # def quantize(A, P, thresh_real=0, thresh_im=0):
        #     mask = np.zeros(np.shape(A), dtype=complex)
        #     mask[:P, :] = (1 / math.sqrt(2)) * (
        #             np.sign(A[:P, :].real - (thresh_real)) + (1j * (np.sign(A[:P, :].imag - ((thresh_im))))))
        #     mask[P:, :] = A[P:, :]
        #     return mask

        eigvals, eigvecs = np.linalg.eig(R)
        sorted_indices = np.argsort(eigvals.real)[::-1]  # Sort eigenvalues in descending order
        eigvecs_sorted = eigvecs[:, sorted_indices]
        En = eigvecs_sorted[:, pram.D:]

        music_spectrum = np.zeros(num_angles)
        for idx, theta in enumerate(theta_range):
            steering_vector = np.exp(-1j * np.pi * np.arange(pram.M) * np.sin(theta))  # TODO- out of the loop
            music_spectrum[idx] = 1 / np.linalg.norm(En.conj().T @ steering_vector)

        peaks, _ = ss.find_peaks(music_spectrum)
        peaks = list(peaks)
        peaks.sort(key=lambda x: music_spectrum[x])
        pred = np.array(peaks[-pram.D:])
        pred = np.sort(pred)[::-1]#np.subtract(np.sort(teta_vector), 90)
        teta_vector[i,:] = pred
        # print("====")
        # print(pred)
        # print(labels[i, :])

    sub_vec_old = teta_vector - labels
    mask = np.logical_and(-pram.C < np.min(sub_vec_old, axis=1), np.max(sub_vec_old, axis=1) < pram.C)
    sub_vec_new = sub_vec_old[mask]
    #print("Relevant tests:",sub_vec_new.shape[0])
    RMSE = ((np.sum(np.sum(np.power(sub_vec_new, 2), 1)) / (sub_vec_new.shape[0] * (teta_vector.shape[1]))) ** 0.5)
    return RMSE # TODO modulo

if __name__ == "__main__":
    print("Not main file")