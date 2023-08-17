from matplotlib import pyplot as plt
from models import *
from train_func import *
from functions_NN import *
from classes_NN import *
import torch
import numpy as np

if __name__ == "__main__":
    SNR_space = np.linspace(-5, 25, 8)
    N_a = [0, 10, 5]
    N_q = [10, 0, 5]
    pram = {"snap":400, "teta_range":[0,60], "D":2, "C":10}
    train_prameters = train_prameters(10000, 100, 100, 20, 0.001)
    my_dict = {"device":"Cuda",
               "Generate new data": False,
               "Train": False, "Test": False}
# ======================================================================================================================
    if my_dict["device"] == "Cuda":
        file_path = '/home/mazya/DNN/'
    else:
        file_path = 'C:/Users/Yaniv/PycharmProjects/DOA/DNN/'

    Error = np.zeros((len(SNR_space), len(N_a)))
    for i in range(len(SNR_space)):
        for j in range(len(N_a)):
            my_parameters = prameters_class(N_a[j]+N_q[j],N_q[j], SNR_space[i], pram["snap"], pram["teta_range"],pram["D"],pram["C"]) #M=N_a+N_q
            if my_dict["Generate new data"]:
                generate_data(my_parameters,train_prameters,file_path)

            data_file_path = file_path+'Data/'
            my_data = My_data(data_file_path,my_parameters)

            if my_dict["Train"]:
                my_model = CNN(my_parameters)
                my_model.weight_init(mean=0, std=0.02)
                my_train(my_data, my_model,my_parameters, train_prameters, 'Trained_Model/',True)

            elif my_dict["Test"]:
                Model = CNN(my_parameters)
                Model.load_state_dict(torch.load(file_path+'Trained_Model/'
                                                 +f'trained_model_N_a={my_parameters.M-my_parameters.N_q}_N_q={my_parameters.N_q}_SNR={my_parameters.SNR}.pth'))
                Model.eval()
                Error[i, j] = test_model(Model,my_data.data_test,my_data.labels_test,my_parameters.C)

    fig = plt.figure(figsize=(10, 6))
    for i in range(len(N_a)):
        plt.plot(SNR_space, Error[:, i], label=f'Analog={N_a[i]}, Quantize={N_q[i]}')
    plt.title(f"RMSE for snap={my_parameters.snap}, M={my_parameters.M}, D={my_parameters.D}")
    plt.ylabel("RMSE (Deg.)")
    plt.xlabel("SNR [dB]")
    plt.legend()
    plt.show()