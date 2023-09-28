import numpy as np
import math
import scipy.signal as ss
from numpy import linalg as LA
from functions import covariance, quantize, observ,covariance_matrix
from classes import Matrix_class, prameters_class
from matplotlib import pyplot as plt


def music_algorithm(pram):
    theta_range = np.radians(np.arange(pram.teta_range[0], pram.teta_range[1], pram.Res))  # Convert angles to radians
    num_angles = len(theta_range)
    rho = pram.D * (10 ** (-pram.SNR / 10) + 1)
    labels = np.zeros((pram.monte, pram.D))
    teta_vector1 = np.zeros((pram.monte, pram.D))
    teta_vector2 = np.zeros((pram.monte, pram.D))
    for i in range(pram.monte):
        while True:
            if pram.D == 1:
                teta = np.random.randint(pram.teta_range[0]+1, pram.teta_range[1]-1, size=pram.D) #ss.find_peaks
                # doesnt find edges/limit points
            else:
                while True:
                    teta = np.random.randint(pram.teta_range[0]+1, pram.teta_range[1]-1, size=pram.D)
                    if abs(teta[0]-teta[1]) >> pram.delta:
                        break
                teta = np.sort(teta)[::-1]
            labels[i, :] = teta
            A = Matrix_class(pram.M, labels[i, :]).matrix()
            my_vec = observ(pram.SNR, pram.snapshot, A)
            my_vec = quantize(my_vec, pram.N_q)
            R = np.cov(my_vec) #covariance(my_vec, my_vec)
            R2 = np.zeros(R.shape, dtype=complex)
            R2[:pram.N_q,:pram.N_q] = rho * (np.sin((math.pi / 2) * R[:pram.N_q,:pram.N_q].real)
                                            + 1j * np.sin((math.pi / 2) * R[:pram.N_q,:pram.N_q].imag)) #R_quantize
            R2[pram.N_q:,:pram.N_q] = ((math.pi*rho/2)**0.5)*R[pram.N_q:,:pram.N_q]#R_mixed
            R2[:pram.N_q,pram.N_q:] = ((math.pi*rho/2)**0.5)*R[:pram.N_q,pram.N_q:]#R_mixed
            R2[pram.N_q:,pram.N_q:] = R[pram.N_q:,pram.N_q:] #R_analog
            pred1 = covariance_matrix(num_angles, theta_range, pram, R)
            pred2 = covariance_matrix(num_angles, theta_range, pram, R2)
            if pred1.shape == teta_vector1[i,:].shape and pred2.shape == teta_vector1[i,:].shape:
                break
        teta_vector1[i,:] = pred1*pram.Res+pram.teta_range[0]
        teta_vector2[i, :] = pred2*pram.Res + pram.teta_range[0]
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
    monte = 3
    delta = 2
    Res = 0.125
    N_a = [0, 0]
    N_q = [10, 5]
    for i in range(len(N_q)):
        my_parameters = prameters_class(N_a[i]+N_q[i],N_q[i],D,teta_range,SNR,snap,monte,delta,Res)
        print(music_algorithm(my_parameters))
