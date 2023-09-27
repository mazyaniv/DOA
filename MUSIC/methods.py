import numpy as np
import math
import scipy.signal as ss
from numpy import linalg as LA
from functions import covariance, quantize, observ,covariance_matrix
from classes import Matrix_class, prameters_class
from matplotlib import pyplot as plt


def music_algorithm(pram,method=0):
    theta_range = np.radians(np.arange(pram.teta_range[0], pram.teta_range[1], 1))  # Convert angles to radians
    num_angles = len(theta_range)
    rho = pram.D * (10 ** (-pram.SNR / 10) + 1)
    labels = np.zeros((pram.monte, pram.D))
    teta_vector1 = np.zeros((pram.monte, pram.D))
    teta_vector2 = np.zeros((pram.monte, pram.D))
    for i in range(pram.monte):
        while True:
            if pram.D == 1:
                teta = np.random.randint(pram.teta_range[0]+1, pram.teta_range[1]-1, size=pram.D) #ss.find_peaks
                # doesnt find limit points
            else:
                while True:
                    teta = np.random.randint(pram.teta_range[0]+1, pram.teta_range[1]-1, size=pram.D)
                    if abs(teta[0]-teta[1]) >> pram.Reso:
                        break
                teta = np.sort(teta)[::-1]
            labels[i, :] = teta
            A = Matrix_class(pram.M, labels[i, :]).matrix()
            my_vec = observ(pram.SNR, pram.snapshot, A)
            my_vec = quantize(my_vec, pram.N_q)
            R = np.cov(my_vec) #covariance(my_vec, my_vec)
            R2 = np.zeros(R.shape, dtype=complex)
            # if method == 1:  #quantized_sin
            #     R[:pram.N_q,:pram.N_q] = rho * (np.sin((math.pi / 2) * R[:pram.N_q,:pram.N_q].real)
            #                                     + 1j * np.sin((math.pi / 2) * R[:pram.N_q,:pram.N_q].imag)) #R_quantize
            #     R[pram.N_q:,:pram.N_q] = ((math.pi*rho/2)**0.5)*R[pram.N_q:,:pram.N_q]#R_mixed
            #     R[:pram.N_q,pram.N_q:] = ((math.pi*rho/2)**0.5)*R[:pram.N_q,pram.N_q:]#R_mixed
            #     R[pram.N_q:,pram.N_q:] = R[pram.N_q:,pram.N_q:] #R_analog
            #
            # elif method == 2:  #quantized_lin
            #     R[:pram.N_q, :pram.N_q] = ((rho * math.pi / 2) *
            #                                (np.subtract(R[:pram.N_q, :pram.N_q], 1 - (2 / math.pi) * np.identity(pram.N_q)))) # R_quantize
            #     R[pram.N_q:, :pram.N_q] = ((math.pi*rho/2) ** 0.5) * R[pram.N_q:, :pram.N_q]  # R_mixed
            #     R[:pram.N_q, pram.N_q:] = ((math.pi*rho/2) ** 0.5) * R[:pram.N_q, pram.N_q:]  # R_mixed
            #     R[pram.N_q:, pram.N_q:] = R[pram.N_q:, pram.N_q:]  # R_analog
            # eigvals, eigvecs = np.linalg.eig(R)
            # sorted_indices = np.argsort(eigvals.real)[::-1]  # Sort eigenvalues in descending order
            # eigvecs_sorted = eigvecs[:, sorted_indices]
            # En = eigvecs_sorted[:, pram.D:]
            #
            # music_spectrum = np.zeros(num_angles)
            # for idx, theta in enumerate(theta_range):
            #     steering_vector = np.exp(-1j * np.pi * np.arange(pram.M) * np.sin(theta))
            #     music_spectrum[idx] = 1 / np.linalg.norm(En.conj().T @ steering_vector)
            # # plt.plot(np.degrees(theta_range), music_spectrum)
            # # plt.title(f"N_a={pram.M-pram.N_q}, N_q={pram.N_q},theta={teta},method={method}")
            # # plt.show()
            #
            # peaks, _ = ss.find_peaks(music_spectrum)
            # peaks = list(peaks)
            # peaks.sort(key=lambda x: music_spectrum[x])
            # pred = np.array(peaks[-pram.D:])
            # pred = np.sort(pred)[::-1]#np.subtract(np.sort(teta_vector), 90)
            R2[:pram.N_q,:pram.N_q] = rho * (np.sin((math.pi / 2) * R[:pram.N_q,:pram.N_q].real)
                                            + 1j * np.sin((math.pi / 2) * R[:pram.N_q,:pram.N_q].imag)) #R_quantize
            R2[pram.N_q:,:pram.N_q] = ((math.pi*rho/2)**0.5)*R[pram.N_q:,:pram.N_q]#R_mixed
            R2[:pram.N_q,pram.N_q:] = ((math.pi*rho/2)**0.5)*R[:pram.N_q,pram.N_q:]#R_mixed
            R2[pram.N_q:,pram.N_q:] = R[pram.N_q:,pram.N_q:] #R_analog
            pred1 = covariance_matrix(num_angles, theta_range, pram, R)
            pred2 = covariance_matrix(num_angles, theta_range, pram, R2)
            if pred1.shape == teta_vector1[i,:].shape and pred2.shape == teta_vector1[i,:].shape:
                break
        teta_vector1[i,:] = pred1+pram.teta_range[0]
        teta_vector2[i, :] = pred2 + pram.teta_range[0]
    sub_vec1 = teta_vector1 - labels
    sub_vec2 = teta_vector2 - labels

    RMSE1 = ((np.sum(np.sum(np.power(sub_vec1, 2), 1)) / (sub_vec1.shape[0] * (teta_vector1.shape[1]))) ** 0.5)
    RMSE2 = ((np.sum(np.sum(np.power(sub_vec2, 2), 1)) / (sub_vec2.shape[0] * (teta_vector2.shape[1]))) ** 0.5)
    return RMSE1, RMSE2 #TODO modulo

if __name__ == "__main__":
    SNR = -5
    snap = 400
    D = 2
    teta_range = [-60, 60]
    monte = 1
    Res = 5

    N_a = [1, 0]#, 10]
    N_q = [0, 5]#, 0]
    me = np.zeros((len(N_q),10, 8),dtype=complex)
    for i in range(len(N_q)):
        my_parameters = prameters_class(N_a[i]+N_q[i],N_q[i],D,teta_range,SNR,snap,monte,Res)
        music_algorithm(my_parameters)
        # music_algorithm(my_parameters,1)