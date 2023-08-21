from methods import music_algorithm
from classes import prameters_class
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    SNR_space = np.linspace(-10, 10, 6)
    #SNR = 0
    #snap_space = np.linspace(100, 600, 5)
    snap = 400

    N_a = [2,8,5,0,10]
    N_q = [8,2,5,10,0]
    D = 2
    teta_range = [0, 60]
    monte = 1000
    C = 10  # Mask
    Error1 = np.zeros((len(SNR_space), len(N_a)))
    Error2 = np.zeros((len(SNR_space), len(N_a)))
    for i in range(len(SNR_space)):
        for j in range(len(N_a)):
            my_parameters = prameters_class(N_a[j]+N_q[j],N_q[j],D,teta_range,SNR_space[i],snap,monte,C)
            if j == len(N_a)-1:
                Error1[i,j] = music_algorithm(my_parameters)
            else:
                Error1[i,j] = music_algorithm(my_parameters)
                Error2[i,j] = music_algorithm(my_parameters,1)

    fig = plt.figure(figsize=(12, 8))
    colors = ['b', 'g', 'orange', 'black','red']
    for i in range(len(N_a)):
        if i > len(N_a)-3:
            style = 'dashed'
        else:
            style = 'solid'
        plt.plot(SNR_space, Error1[:,i],color = colors[i],linestyle=style, label=f'Analog={N_a[i]}, Quantize={N_q[i]}')
        if i < len(N_a)-1:
            plt.plot(SNR_space, Error2[:, i],color = colors[i],linestyle=style,marker='o', label=f'Analog={N_a[i]}, Quantize={N_q[i]}, Sin recon.')
    plt.grid()
    plt.title(f"RMSE for snap={my_parameters.snapshot}, M={my_parameters.M}, D={my_parameters.D}, monte={my_parameters.monte}")
    plt.ylabel("RMSE (Deg.)")
    plt.xlabel("SNR [dB]")
    plt.legend(loc='lower left', fontsize='small')
    plt.show()





