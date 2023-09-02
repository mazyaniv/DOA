import numpy as np
import math
import scipy.signal as ss
from numpy import linalg as LA
from functions import covariance, quantize, observ
from classes import Matrix_class, prameters_class
from matplotlib import pyplot as plt


def music_algorithm(pram,method=0):
    theta_range = np.radians(np.arange(pram.teta_range[0], pram.teta_range[1], 1))  # Convert angles to radians
    num_angles = len(theta_range)
    rho = pram.D * (10 ** (-pram.SNR / 10) + 1)
    labels = np.zeros((pram.monte, pram.D))
    teta_vector = np.zeros((pram.monte, pram.D))
    En_vector = np.zeros((pram.monte, pram.M, pram.M-pram.D),dtype=complex)
    #print(teta_vector.shape)
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
            # print(my_vec)
            # print("=======")
            my_vec = quantize(my_vec, pram.N_q)
            R = np.cov(my_vec) #covariance(my_vec, my_vec)
            # print(my_vec)
            # print("=======")
            # print(R)
            # print("=======")
            if method == 1:  #quantized_sin
                R[:pram.N_q,:pram.N_q] = rho * (np.sin((math.pi / 2) * R[:pram.N_q,:pram.N_q].real)
                                                + 1j * np.sin((math.pi / 2) * R[:pram.N_q,:pram.N_q].imag)) #R_quantize
                R[pram.N_q:,:pram.N_q] = ((math.pi*rho/2)**0.5)*R[pram.N_q:,:pram.N_q]#R_mixed
                R[:pram.N_q,pram.N_q:] = ((math.pi*rho/2)**0.5)*R[:pram.N_q,pram.N_q:]#R_mixed
                R[pram.N_q:,pram.N_q:] = R[pram.N_q:,pram.N_q:] #R_analog

            elif method == 2:  #quantized_lin
                R[:pram.N_q, :pram.N_q] = ((rho * math.pi / 2) *
                                           (np.subtract(R[:pram.N_q, :pram.N_q], 1 - (2 / math.pi) * np.identity(pram.N_q)))) # R_quantize
                R[pram.N_q:, :pram.N_q] = ((math.pi*rho/2) ** 0.5) * R[pram.N_q:, :pram.N_q]  # R_mixed
                R[:pram.N_q, pram.N_q:] = ((math.pi*rho/2) ** 0.5) * R[:pram.N_q, pram.N_q:]  # R_mixed
                R[pram.N_q:, pram.N_q:] = R[pram.N_q:, pram.N_q:]  # R_analog

            eigvals, eigvecs = np.linalg.eig(R)
            sorted_indices = np.argsort(eigvals.real)[::-1]  # Sort eigenvalues in descending order
            eigvecs_sorted = eigvecs[:, sorted_indices]
            # print("N_a:", pram.M-pram.N_q,"N_q:", pram.N_q)
            # print(eigvals.real[sorted_indices])
            # print("============")
            # print(eigvecs_sorted)

            En = eigvecs_sorted[:, pram.D:]
            En_vector[i,:,:] = En

            music_spectrum = np.zeros(num_angles)
            for idx, theta in enumerate(theta_range):
                steering_vector = np.exp(-1j * np.pi * np.arange(pram.M) * np.sin(theta))
                music_spectrum[idx] = 1 / np.linalg.norm(En.conj().T @ steering_vector)

            peaks, _ = ss.find_peaks(music_spectrum)
            peaks = list(peaks)
            peaks.sort(key=lambda x: music_spectrum[x])
            pred = np.array(peaks[-pram.D:])
            pred = np.sort(pred)[::-1]#np.subtract(np.sort(teta_vector), 90)
            if pred.shape == teta_vector[i,:].shape:
                break
        teta_vector[i,:] = pred+pram.teta_range[0]
    #En = np.mean(En_vector, 0)
    #print(LA.norm(En, "fro"))
    # LA.norm(cov_matrix, "fro")
    sub_vec = teta_vector - labels
    # print("real value:", labels)
    # print("estimator:", teta_vector)
    # print("sub:", sub_vec)
    # print("============")

    RMSE = ((np.sum(np.sum(np.power(sub_vec, 2), 1)) / (sub_vec.shape[0] * (teta_vector.shape[1]))) ** 0.5)
    #print("RMSE:", RMSE)
    return RMSE #TODO modulo

if __name__ == "__main__":
    SNR = -1
    snap = 400
    D = 2
    teta_range = [0, 60]
    monte = 500
    C = 5  # Res

    N_a = [0,2,5,8,10]
    N_q = [10,8,5,2,0]
    me = np.zeros((len(N_a),10, 8),dtype=complex)
    for i in range(len(N_a)):
        my_parameters = prameters_class(N_a[i]+N_q[i],N_q[i],D,teta_range,SNR,snap,monte,C)
        music_algorithm(my_parameters)
        music_algorithm(my_parameters,1)