import numpy as np
import math
import scipy.signal as ss
from numpy import linalg as LA
from functions import quantize, observ,angles_generate,music,root_music,esprit
from classes import Matrix_class, prameters_class
from matplotlib import pyplot as plt


def general(pram):
    rho = pram.D*(10**(pram.SNR / 10))+1#pram.D * (10 ** (-pram.SNR / 10)+1)
    labels = np.zeros((pram.monte, pram.D))
    teta_vector1 = np.zeros((pram.monte, pram.D))
    teta_vector2 = np.zeros((pram.monte, pram.D))
    for i in range(pram.monte):
        while True:
            teta = angles_generate(pram)
            labels[i, :] = teta

            A = Matrix_class(pram.M, labels[i, :]).matrix()
            my_vec = observ(pram.SNR, pram.snapshot, A)
            my_vec = quantize(my_vec, pram.N_q)
            R = np.cov(my_vec) #covariance(my_vec, my_vec)

            R1 = np.zeros(R.shape, dtype=complex)
            R1[:pram.N_q, :pram.N_q] = rho*((math.pi / 2) *
                                       (np.subtract(R[:pram.N_q, :pram.N_q],
                                                    (1 - (2 / math.pi)) * np.identity(pram.N_q)))) # R_quantize_lin
            R1[pram.N_q:, :pram.N_q] = ((math.pi*rho/2)**0.5)*R[pram.N_q:, :pram.N_q]  # R_mixed
            R1[:pram.N_q, pram.N_q:] = ((math.pi*rho/2)**0.5)*R[:pram.N_q, pram.N_q:]  # R_mixed
            R1[pram.N_q:, pram.N_q:] = R[pram.N_q:, pram.N_q:]  # R_analog

            R2 = np.zeros(R.shape, dtype=complex)
            R2[:pram.N_q,:pram.N_q] = rho*(np.sin((math.pi / 2) * R[:pram.N_q,:pram.N_q].real)
                                            + 1j * np.sin((math.pi / 2) * R[:pram.N_q,:pram.N_q].imag)) #R_quantize_sin
            R2[pram.N_q:,:pram.N_q] = ((math.pi*rho/2)**0.5)*R[pram.N_q:,:pram.N_q]#R_mixed
            R2[:pram.N_q,pram.N_q:] = ((math.pi*rho/2)**0.5)*R[:pram.N_q,pram.N_q:]#R_mixed
            R2[pram.N_q:,pram.N_q:] = R[pram.N_q:,pram.N_q:] #R_analog
            if pram.dictio['MUSIC'] == 1:
                pred1 = music(pram, R)
                pred2 = music(pram, R2)
            elif pram.dictio['Root-MUSIC'] == 1:
                pred1 = root_music(pram, R)
                pred2 = root_music(pram, R2)
            elif pram.dictio['ESPRIT'] == 1:
                pred1 = esprit(pram, R)
                pred2 = esprit(pram, R2)
            if pred1.shape == teta_vector1[i,:].shape and pred2.shape == teta_vector1[i,:].shape:
                break
        teta_vector1[i,:] = pred1
        teta_vector2[i, :] = pred2
    sub_vec1 = teta_vector1 - labels
    sub_vec2 = teta_vector2 - labels
    RMSE1 = ((np.sum(np.sum(np.power(sub_vec1, 2), 1)) / (sub_vec1.shape[0] * (teta_vector1.shape[1]))) ** 0.5)
    RMSE2 = ((np.sum(np.sum(np.power(sub_vec2, 2), 1)) / (sub_vec2.shape[0] * (teta_vector2.shape[1]))) ** 0.5)
    return RMSE1, RMSE2 #TODO modulo
def detect(pram):
    teta = [20+pram.delta, 20]
    count1 = 0
    count2 = 0
    label = np.array(teta)
    rho = pram.D*(10**(pram.SNR / 10))+1#pram.D * (10 ** (-pram.SNR / 10) + 1)
    teta_vector1 = np.zeros((pram.monte, pram.D))
    for i in range(pram.monte):
        while True:
            A = Matrix_class(pram.M, label).matrix()
            my_vec = observ(pram.SNR, pram.snapshot, A)
            my_vec = quantize(my_vec, pram.N_q)
            R = np.cov(my_vec) #covariance(my_vec, my_vec)
            R2 = np.zeros(R.shape, dtype=complex)
            R2[:pram.N_q,:pram.N_q] = rho * (np.sin((math.pi / 2) * R[:pram.N_q,:pram.N_q].real)
                                            + 1j * np.sin((math.pi / 2) * R[:pram.N_q,:pram.N_q].imag)) #R_quantize
            R2[pram.N_q:,:pram.N_q] = ((math.pi*rho/2)**0.5)*R[pram.N_q:,:pram.N_q]#R_mixed
            R2[:pram.N_q,pram.N_q:] = ((math.pi*rho/2)**0.5)*R[:pram.N_q,pram.N_q:]#R_mixed
            R2[pram.N_q:,pram.N_q:] = R[pram.N_q:,pram.N_q:] #R_analog
            if pram.dictio['MUSIC'] == 1:
                pred1 = music(pram, R2)
                pred2 = music(pram, R2)
            elif pram.dictio['Root-MUSIC'] == 1:
                pred1 = root_music(pram, R)
                pred2 = root_music(pram, R2)
            elif pram.dictio['ESPRIT'] == 1:
                pred1 = esprit(pram, R)
                pred2 = esprit(pram, R2)
            if pred1.shape == teta_vector1[i,:].shape and pred2.shape == teta_vector1[i,:].shape:
                break
        if (abs(pred1-label)<pram.delta/2).all():
            count1 += 1
        if (abs(pred2 - label) < pram.delta / 2).all():
            count2 += 1
    return count1/pram.monte, count2/pram.monte
def norm(pram):
    rho = pram.D*(10**(pram.SNR / 10))+1#pram.D * (10 ** (-pram.SNR / 10)+1)
    norm_vec_R = np.zeros(pram.monte)
    norm_vec_R_a = np.zeros(pram.monte)
    norm_vec_R2 = np.zeros(pram.monte)
    for i in range(pram.monte):
        teta = angles_generate(pram)
        A = Matrix_class(pram.M, teta).matrix()
        my_vec = observ(pram.SNR, pram.snapshot, A)
        R_a = np.cov(my_vec) #covariance(my_vec, my_vec)
        my_vec = quantize(my_vec, pram.N_q)
        R = np.cov(my_vec) #covariance(my_vec, my_vec)

        R2 = np.zeros(R.shape, dtype=complex)
        R2[:pram.N_q,:pram.N_q] = rho*(np.sin((math.pi / 2) * R[:pram.N_q,:pram.N_q].real)
                                        + 1j * np.sin((math.pi / 2) * R[:pram.N_q,:pram.N_q].imag)) #R_quantize_sin
        R2[pram.N_q:,:pram.N_q] = ((math.pi*rho/2)**0.5)*R[pram.N_q:,:pram.N_q]#R_mixed
        R2[:pram.N_q,pram.N_q:] = ((math.pi*rho/2)**0.5)*R[:pram.N_q,pram.N_q:]#R_mixed
        R2[pram.N_q:,pram.N_q:] = R[pram.N_q:,pram.N_q:] #R_analog
        norm_vec_R_a[i], norm_vec_R[i], norm_vec_R2[i] = LA.norm(R_a, "fro"), LA.norm(R, "fro"), LA.norm(R2, "fro")
    return np.mean(norm_vec_R_a), np.mean(norm_vec_R), np.mean(norm_vec_R2)

if __name__ == "__main__":
    N_a = 0  # [0, 0]
    N_q = 2  # [10, 5]
    SNR = -10
    snap = 1000
    D = 2
    teta_range = [-60, 60]
    monte = 1
    delta = 2
    Res = 0.0625
    method_dict = {'MUSIC': 0, 'Root-MUSIC': 1, 'ESPRIT': 0}

    my_parameters = prameters_class(N_a + N_q, N_q, SNR, snap, D, teta_range, monte, delta, Res, method_dict)
    RMSE1, RMSE2 = general(my_parameters)
