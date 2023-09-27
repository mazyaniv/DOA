from methods import music_algorithm
from classes import prameters_class
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    SNR_space = np.linspace(-5, 10, 5)
    # SNR = 0
    # snap_space = np.linspace(100, 600, 5)
    snap = 200
    N_a = [0,1,10]
    N_q = [10,9,0]
    D = 2
    teta_range = [-60, 60]
    monte = 300
    Res = 2
    Error1 = np.zeros((len(SNR_space), len(N_a)))
    Error2 = np.zeros((len(SNR_space), len(N_a)))
    for i in range(len(SNR_space)):
        for j in range(len(N_a)):
            my_parameters = prameters_class(N_a[j]+N_q[j],N_q[j],D,teta_range,SNR_space[i],snap,monte,Res)
            # if j == len(N_a)-1:
            #     Error1[i,j] = music_algorithm(my_parameters)
            #     Error2[i,j] = music_algorithm(my_parameters,1)
            # else:
            #     Error1[i,j] = music_algorithm(my_parameters)
            #     Error2[i,j] = music_algorithm(my_parameters,1)
            Error1[i, j], Error2[i, j] = music_algorithm(my_parameters)
    fig = plt.figure(figsize=(12, 8))
    colors = ['b', 'g', 'orange', 'black','red']
    for i in range(len(N_a)):
        # if i > len(N_a)-3:
        #     style = 'dashed'
        # else:
        #     style = 'solid'
        plt.plot(SNR_space, Error1[:,i],color = colors[i],linestyle='solid', label=f'Quantize={N_q[i]}, Analog={N_a[i]}')
        if i < len(N_a)-1:
            plt.plot(SNR_space, Error2[:, i], color=colors[i], linestyle='dashed',
                     label=f'Quantize={N_q[i]}, Analog={N_a[i]}, Sin recon.')
        # if i < len(N_a)-1:
        #     plt.plot(SNR_space, Error2[:, i],color = colors[i],linestyle='dashed', label=f'Quantize={N_q[i]}, Analog={N_a[i]}, Sin recon.')
    plt.grid()
    plt.title(f"RMSE for snap={my_parameters.snapshot}, M={my_parameters.M}, D={my_parameters.D}, monte={my_parameters.monte}, Res={my_parameters.Reso}")
    # plt.yscale('log')
    plt.ylabel("RMSE (Deg.)")
    plt.xlabel("SNR [dB]")
    plt.legend(loc='upper right', fontsize='small')
    plt.show()





