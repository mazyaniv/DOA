from methods import general
from classes import prameters_class
import numpy as np
from matplotlib import pyplot as plt
from functions import get_key_by_value

if __name__ == "__main__":
    N_a = [0, 1, 10]
    N_q = [10, 9, 0]
    SNR_space = np.linspace(-10, 10, 4)
    snap = 200
    # snap_space = np.linspace(100, 600, 5)
    # SNR = 0
    D = 2
    teta_range = [-60, 60]
    monte = 10
    delta = 1 #Minimum gap between two determenistic angles
    Res = 0.5
    method_dict = {'MUSIC': 1, 'Root-MUSIC': 0, 'ESPRIT': 0}

    Error1 = np.zeros((len(SNR_space), len(N_a)))
    Error2 = np.zeros((len(SNR_space), len(N_a)))
    for i in range(len(SNR_space)):
        for j in range(len(N_a)):
            my_parameters = prameters_class(N_a[j]+N_q[j],N_q[j],SNR_space[i],snap,D,teta_range,monte,delta,Res,method_dict)
            Error1[i, j], Error2[i, j] = general(my_parameters)
    fig = plt.figure(figsize=(12, 8))
    colors = ['red', 'b', 'black']
    for i in range(len(N_a)):
        plt.plot(SNR_space, Error1[:,i],color = colors[i],linestyle='solid', label=f'Quantize={N_q[i]}, Analog={N_a[i]}')
        if i < len(N_a)-1:
            plt.plot(SNR_space, Error2[:, i], color=colors[i], linestyle='dashed',
                     label=f'Quantize={N_q[i]}, Analog={N_a[i]}, Sin recon.')
        # if i < len(N_a)-1:
        #     plt.plot(SNR_space, Error2[:, i],color = colors[i],linestyle='dashed', label=f'Quantize={N_q[i]}, Analog={N_a[i]}, Sin recon.')
    value = get_key_by_value(method_dict, 1)
    plt.grid()
    plt.title(f"RMSE for Snap={my_parameters.snapshot}, Monte={my_parameters.monte}, Res={my_parameters.Res}, Delta={my_parameters.delta}, "
              f"Method: {value}")
    plt.ylabel("RMSE (Deg.)")
    plt.xlabel("SNR [dB]")
    plt.legend(loc='upper right', fontsize='small')
    plt.show()




