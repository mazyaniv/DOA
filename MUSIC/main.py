from methods import general, detect, norm
from classes import prameters_class
import numpy as np
from matplotlib import pyplot as plt
from functions import get_key_by_value

if __name__ == "__main__":
    N_a = [0, 2, 20]
    N_q = [20, 18, 0]
    D = 2
    teta_range = [-60, 60]
    # SNR = 2
    SNR_space = np.linspace(-10, 10, 5)
    snap = 150
    # snap_space = np.linspace(100, 1000, 10, dtype=int)
    monte = 100
    delta = 5 #Minimal gap between two determenistic angles
    Res = 0.5
    method_dict = {'MUSIC': 1, 'Root-MUSIC': 0, 'ESPRIT': 0}
    # delta_space = np.linspace(0.8, 6, 20)
    relevant_space = SNR_space #TODO

    Error1 = np.zeros((len(relevant_space), len(N_q)))
    Error2 = np.zeros((len(relevant_space), len(N_q)))
    # Error3 = np.zeros((len(relevant_space), len(N_q)))
    for i in range(len(relevant_space)):
        for j in range(len(N_q)):
            my_parameters = prameters_class(N_a[j]+N_q[j],N_q[j],relevant_space[i],snap,D,teta_range,monte,delta,
                                            Res,method_dict) #TODO
            Error1[i, j], Error2[i, j] = general(my_parameters) #TODO
            # Error1[i, j], Error2[i, j],Error3[i,j]  = norm(my_parameters)
    fig = plt.figure(figsize=(12, 8))
    colors = ['red', 'b', 'black']
    for i in range(len(N_q)):#TODO
        plt.plot(relevant_space, Error1[:,i],linestyle='solid',marker=".",color=colors[i], label=f'N_a={N_a[i]},N_q={N_q[i]}')
        if i < len(N_q)-1:
            plt.plot(relevant_space, Error2[:, i], linestyle='dashed',marker=".",color=colors[i],
                     label=f'N_a={N_a[i]},N_q={N_q[i]}, Sin recon.')
    # for i in range(len(N_q)):
    #     # plt.plot(relevant_space, Error1[:,i],linestyle='solid',color=colors[i], label=f'N_a={N_a[i]},N_q={N_q[i]}, R_analog')
    #     plt.plot(relevant_space, Error2[:, i], linestyle='solid',marker="*", color='red',
    #              label=f'N_a={N_a[i]},N_q={N_q[i]}, R')
    #     plt.plot(relevant_space, Error2[:, i], linestyle='dashed',marker=".",color='blue',
    #              label=f'N_a={N_a[i]},N_q={N_q[i]}, $\hat R$')

    value = get_key_by_value(method_dict, 1)
    plt.grid()
    plt.title(f"RMSE for SNR={my_parameters.SNR}, Monte={my_parameters.monte}, Res={my_parameters.Res}, "
              f"snap={my_parameters.snapshot}, Method: {value}") #TODO
    plt.ylabel("RMSE")
    # plt.ylabel("$|A|_F$")
    # plt.yscale('log')
    # plt.xlabel("snapshots") #TODO
    plt.xlabel("SNR[dB]")
    # plt.ylabel("Resolution Probability")
    # plt.xlabel("$\Delta^\degree$")
    plt.legend(loc='upper right', fontsize='small')
    plt.show()




