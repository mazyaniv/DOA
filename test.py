from methods import music_algorithm
from classes import prameters_class
import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    SNR_space = np.linspace(-10, 10, 7)
    Error = np.zeros((len(SNR_space),4))
    D = 2
    teta_range = [0, 60]
    snap = 500
    monte = 10
    C = 10 #Mask
    N_a = [1, 1, 2, 2]
    N_q = [10, 15, 10, 15]
    for i in range(len(SNR_space)):
        for j in range(len(N_a)):
            my_parameters = prameters_class(N_a[j]+N_q[j],N_q[j],D,teta_range,SNR_space[i],snap,monte,C)
            Error[i,j] = music_algorithm(my_parameters)
        # parameters_q = prameters_class(10,10,D, teta_range, SNR_space[i],snap, monte, C)
        # Error_q[i] = music_algorithm(parameters_q)

    fig = plt.figure(figsize=(10, 6))
    for i in range(4):
        plt.plot(SNR_space, Error[:,i], label="Analog")
        #plt.plot(SNR_space, Error, label="Quantize")
    plt.ylabel("RMSE (Deg.)")
    plt.xlabel("SNR [dB]")
    plt.legend()
    plt.show()





