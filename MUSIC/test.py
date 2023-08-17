from methods import music_algorithm
from classes import prameters_class
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    SNR_space = np.linspace(-5, 25, 7)
    N_a = [0,10,5]
    N_q = [10,0,5]
    D = 2
    teta_range = [0, 60]
    snap = 10
    monte = 10
    C = 10  # Mask
    Error = np.zeros((len(SNR_space), len(N_a)))
    for i in range(len(SNR_space)):
        for j in range(len(N_a)):
            my_parameters = prameters_class(N_a[j]+N_q[j],N_q[j],D,teta_range,SNR_space[i],snap,monte,C)
            Error[i,j] = music_algorithm(my_parameters)

    fig = plt.figure(figsize=(10, 6))
    for i in range(len(N_a)):
        plt.plot(SNR_space, Error[:,i], label=f'Analog={N_a[i]}, Quantize={N_q[i]}')
    plt.title(f"RMSE for snap={my_parameters.snap}, M={my_parameters.M}, D={my_parameters.D}")
    plt.ylabel("RMSE (Deg.)")
    plt.xlabel("SNR [dB]")
    plt.legend()
    plt.show()





