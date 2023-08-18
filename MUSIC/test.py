from methods import music_algorithm
from classes import prameters_class
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    #SNR_space = np.linspace(-15, 15, 6)
    SNR = 0
    snap_space = np.linspace(100, 1000, 7)
    #snap = 400

    N_a = [1,1,2,2]
    N_q = [8,10,8,10]
    D = 2
    teta_range = [0, 60]
    monte = 200
    C = 10  # Mask
    Error = np.zeros((len(snap_space), len(N_a)))
    for i in range(len(snap_space)):
        for j in range(len(N_a)):
            my_parameters = prameters_class(N_a[j]+N_q[j],N_q[j],D,teta_range,SNR,int(snap_space[i]),monte,C)
            Error[i,j] = music_algorithm(my_parameters)

    fig = plt.figure(figsize=(10, 6))
    for i in range(len(N_a)):
        plt.plot(snap_space, Error[:,i], label=f'Analog={N_a[i]}, Quantize={N_q[i]}')
    plt.title(f"RMSE for SNR={my_parameters.SNR}, M={my_parameters.M}, D={my_parameters.D}, monte={my_parameters.monte}")
    plt.ylabel("RMSE (Deg.)")
    plt.xlabel("snapshots")
    plt.legend()
    plt.show()





