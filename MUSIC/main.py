from methods import general, detect, norm
from classes import prameters_class
import numpy as np
from matplotlib import pyplot as plt
from functions import get_key_by_value
from classes import Matrix_class
from functions import observ, quantize
import numpy.linalg as LA

if __name__ == "__main__":
    N_a = [0,1,10]
    N_q = [10,9,0]
    D = 2
    teta_range = [-60, 60]
    SNR = 0
    # SNR_space = np.linspace(-10, 10, 15)
    snap = [1100,900,500]
    #snap_space = np.linspace(100, 1000, 10)
    monte = 410
    delta = 10 #Minimal gap between two determenistic angles
    Res = 0.1
    method_dict = {'MUSIC': 0, 'Root-MUSIC': 0, 'ESPRIT': 1}
    delta_space = np.linspace(0.8, 6, 15)
    relevant_space = delta_space #TODO


    Error = np.zeros((len(relevant_space), len(N_q)))
    Error1 = np.zeros((len(relevant_space), len(N_q)))
    Error2 = np.zeros((len(relevant_space), len(N_q)))
    for i in range(len(relevant_space)):
        for j in range(len(N_q)):
            my_parameters = prameters_class(N_a[j]+N_q[j],N_q[j],SNR,snap[j],D,teta_range,monte,relevant_space[i],
                                            Res,method_dict) #TODO
            Error[i, j],Error1[i, j] = detect(my_parameters) #TODO #Error2[i, j]
            # Error1[i, j], Error2[i, j],Error3[i,j]  = norm(my_parameters)
    fig = plt.figure(figsize=(12, 8))
    colors = ['red','b', 'black']
    for i in range(len(N_q)):#TODO
        plt.plot(relevant_space, Error[:,i],linestyle='solid',color=colors[i], label=f'N_a={N_a[i]},N_q={N_q[i]}, snap={snap[i]}')
        if i < len(N_q)-1:
            plt.plot(relevant_space, Error1[:, i], linestyle='dashed', marker="x", color=colors[i],
                     label=f'N_a={N_a[i]},N_q={N_q[i]}, Bussgang recon.')
            # plt.plot(relevant_space, Error2[:, i], linestyle='-.', marker="o", color=colors[i],
            #          label=f'N_a={N_a[i]},N_q={N_q[i]}, Filter')
    # for i in range(len(N_q)):
    #     # plt.plot(relevant_space, Error1[:,i],linestyle='solid',color=colors[i], label=f'N_a={N_a[i]},N_q={N_q[i]}, R_analog')
    #     plt.plot(relevant_space, Error2[:, i], linestyle='solid',marker="*", color='red',
    #              label=f'N_a={N_a[i]},N_q={N_q[i]}, R')
    #     plt.plot(relevant_space, Error2[:, i], linestyle='dashed',marker=".",color='blue',
    #              label=f'N_a={N_a[i]},N_q={N_q[i]}, $\hat R$')

    value = get_key_by_value(method_dict, 1)
    ax = plt.gca()
    ax.set_xticks(np.arange(relevant_space[0], relevant_space[-1], 0.25), minor=True)
    ax.grid(which='major', alpha=0.5)
    ax.grid(which='minor', linestyle="--", alpha=0.25)
    # plt.title(f"RMSE for Snap={my_parameters.snapshot}, Monte={my_parameters.monte}, "
    #           f"Method: {value}") #TODO
    # plt.xlabel("snapshots") #TODO
    plt.xlabel("$\Delta^\degree$")
    # plt.xlabel("$SNR_{[dB]}$")
    plt.ylabel("RMSE")
    # plt.ylabel("$|A|_F$")
    # plt.ylabel("Resolution Probability")
    # plt.yscale('log')
    plt.legend(loc='lower left', fontsize='small')
    plt.show()




