from methods import general, detect
from classes import prameters_class
import numpy as np
from matplotlib import pyplot as plt
from functions import get_key_by_value

if __name__ == "__main__":
    N_a = [0, 1, 10]
    N_q = [10, 9, 0]
    D = 2
    teta_range = [-60, 60]
    # SNR_space = np.linspace(-10, 10, 4)
    snap = 300
    # snap_space = np.linspace(100, 1000, 10)
    SNR = 0
    monte = 500
    delta = 5 #Minimal gap between two determenistic angles
    delta_space = np.linspace(1, 6, 15)
    relevant_space = delta_space
    Res = 0.5
    method_dict = {'MUSIC': 0, 'Root-MUSIC': 0, 'ESPRIT': 1}

    Error1 = np.zeros((len(relevant_space), len(N_a)))
    Error2 = np.zeros((len(relevant_space), len(N_a)))
    for i in range(len(relevant_space)):
        for j in range(len(N_a)):
            my_parameters = prameters_class(N_a[j]+N_q[j],N_q[j],SNR,snap,D,teta_range,monte,relevant_space[i],Res,method_dict)
            Error1[i, j], Error2[i, j] = detect(my_parameters)
    fig = plt.figure(figsize=(12, 8))
    colors = ['red', 'b', 'black']
    for i in range(len(N_a)):
        plt.plot(relevant_space, Error1[:,i],color = colors[i],linestyle='solid', label=f'Quantize={N_q[i]}, Analog={N_a[i]}')
        if i < len(N_a)-1:
            plt.plot(relevant_space, Error2[:, i], color=colors[i], linestyle='dashed',
                     label=f'Quantize={N_q[i]}, Analog={N_a[i]}, Sin recon.')
    value = get_key_by_value(method_dict, 1)
    plt.grid()
    plt.title(f"RMSE for SNR={my_parameters.SNR}, Monte={my_parameters.monte}, Res={my_parameters.Res}, Snap={my_parameters.snapshot}, "
              f"Method: {value}")
    plt.ylabel("Resolution Probability")
    plt.xlabel("$\Delta^\degree$")
    plt.legend(loc='upper right', fontsize='small')
    plt.show()




