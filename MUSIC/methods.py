import numpy as np
import math
import scipy.signal as ss
from numpy import linalg as LA
from functions import covariance, quantize, observ
from classes import Matrix_class

def music_algorithm(pram,method=0):
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
        while True:
            print("Back to start")

            A = Matrix_class(pram.M, labels[i, :]).matrix()
            my_vec = observ(pram.SNR, pram.snapshot, A)
            my_vec = quantize(my_vec, pram.N_q)
            R = covariance(my_vec, my_vec)
            if method == 1:  #quantized_sin
                R[:pram.N_q,:pram.N_q] = rho * (np.sin((math.pi / 2) * R[:pram.N_q,:pram.N_q].real)
                                                + 1j * np.sin((math.pi / 2) * R[:pram.N_q,:pram.N_q].imag)) #R_quantize
                R[pram.N_q:,:pram.N_q] = ((math.pi/2)**0.5)*R[pram.N_q:,:pram.N_q]#R_mixed
                R[:pram.N_q,pram.N_q:] = ((math.pi/2)**0.5)*R[:pram.N_q,pram.N_q:]#R_mixed
                R[pram.N_q:,pram.N_q:] = R[pram.N_q:,pram.N_q:] #R_analog

            elif method == 2:  #quantized_lin
                R[:pram.N_q, :pram.N_q] = ((rho * math.pi / 2) *
                                           (np.subtract(R[:pram.N_q, :pram.N_q], 1 - (2 / math.pi) * np.identity(pram.N_q)))) # R_quantize
                R[pram.N_q:, :pram.N_q] = ((math.pi / 2) ** 0.5) * R[pram.N_q:, :pram.N_q]  # R_mixed
                R[:pram.N_q, pram.N_q:] = ((math.pi / 2) ** 0.5) * R[:pram.N_q, pram.N_q:]  # R_mixed
                R[pram.N_q:, pram.N_q:] = R[pram.N_q:, pram.N_q:]  # R_analog

            eigvals, eigvecs = np.linalg.eig(R)
            sorted_indices = np.argsort(eigvals.real)[::-1]  # Sort eigenvalues in descending order
            eigvecs_sorted = eigvecs[:, sorted_indices]
            En = eigvecs_sorted[:, pram.D:]

            music_spectrum = np.zeros(num_angles)
            for idx, theta in enumerate(theta_range):
                steering_vector = np.exp(-1j * np.pi * np.arange(pram.M) * np.sin(theta))
                music_spectrum[idx] = 1 / np.linalg.norm(En.conj().T @ steering_vector)

            peaks, _ = ss.find_peaks(music_spectrum)
            peaks = list(peaks)
            peaks.sort(key=lambda x: music_spectrum[x])
            pred = np.array(peaks[-pram.D:])
            pred = np.sort(pred)[::-1]#np.subtract(np.sort(teta_vector), 90)

            print(pred)

            if pred.shape == teta_vector[i,:].shape:
                break

        print("====")
        teta_vector[i,:] = pred

    sub_vec_old = teta_vector - labels
    mask = np.logical_and(-pram.C < np.min(sub_vec_old, axis=1), np.max(sub_vec_old, axis=1) < pram.C)
    sub_vec_new = sub_vec_old[mask]
    RMSE = ((np.sum(np.sum(np.power(sub_vec_new, 2), 1)) / (sub_vec_new.shape[0] * (teta_vector.shape[1]))) ** 0.5)
    #print("Relevant tests:",sub_vec_new.shape[0])
    print('RMSE = ', RMSE)
    return RMSE #TODO modulo

if __name__ == "__main__":
    from classes import prameters_class
    import numpy as np
    from matplotlib import pyplot as plt

    #print("Not main file")
    SNR_space = np.linspace(-10, 10, 6)
    # SNR = 0
    # snap_space = np.linspace(100, 600, 5)
    snap = 400

    N_a = [2, 8, 5, 0]
    N_q = [8, 2, 5, 10]
    D = 2
    teta_range = [0, 60]
    monte = 200
    C = 10  # Mask
    Error1 = np.zeros((len(SNR_space), len(N_a)))
    Error2 = np.zeros((len(SNR_space), len(N_a)))
    Error3 = np.zeros((len(SNR_space), len(N_a)))
    for i in range(len(SNR_space)):
        for j in range(len(N_a)):
            my_parameters = prameters_class(N_a[j] + N_q[j], N_q[j], D, teta_range, SNR_space[i], snap, monte, C)
            Error1[i, j] = music_algorithm(my_parameters)