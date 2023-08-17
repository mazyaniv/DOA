import torch
import torch.nn as nn
import numpy as np
import math
from matplotlib import pyplot as pltfrom
from functions_NN import *
from classes_NN import *
from models import *
from train_func import *

if __name__ == "__main__":
    my_dict = {"Checking device": False,"Save data":False,"Load data": True, "Train": True, "Test": False}

    file_path = '/home/mazya/DNN/' #'C:/Users/Yaniv/PycharmProjects/DOA/DNN/'
    if my_dict["Checking device"]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)

    my_parameters = prameters_class(10, 0, 2, [0,60], 0, 400)
    train_prameters = train_prameters(1000, 50, 100, 20, 0.001)
    if my_dict["Save data"]:
        data = np.zeros((2, train_prameters.N, my_parameters.M, my_parameters.M), dtype=np.complex128)
        labels = np.zeros((train_prameters.N, my_parameters.D))
        for i in range(0, train_prameters.N):
            if my_parameters.D == 1:
                teta = np.random.randint(my_parameters.teta_range[0], my_parameters.teta_range[1], size=my_parameters.D)
            else:
                while True:
                    teta = np.random.randint(my_parameters.teta_range[0], my_parameters.teta_range[1], size=my_parameters.D)
                    if teta[0] != teta[1]:
                        break
                teta = np.sort(teta)[::-1]
            labels[i, :] = teta
            Observ = quantize_part(observ(teta, my_parameters.M, my_parameters.D,my_parameters.SNR,my_parameters.snap),
                                   my_parameters.P) #Quantize
            R = np.cov(Observ)
            data[0, i, :, :] = np.triu(R, k=1).real
            data[1, i, :, :] = np.triu(R, k=1).imag

        data_train = data[:, :-train_prameters.test_size, :, :]
        labels_train = labels[:-train_prameters.test_size, :]
        data_test = data[:, -train_prameters.test_size:, :, :]
        labels_test = labels[-train_prameters.test_size:, :]

        np.save(file_path+'Data/'+'data_train.npy',data_train)
        np.save(file_path+'Data/'+'labels_train.npy', labels_train)
        np.save(file_path+'Data/'+'data_test.npy', data_test)
        np.save(file_path+'Data/'+'labels_test.npy', labels_test)

    elif my_dict["Load data"]:
        data_train = np.load(file_path+'Data/'+'data_train.npy')
        labels_train = np.load(file_path+'Data/'+'labels_train.npy')
        data_test = np.load(file_path+'Data/'+'data_test.npy')
        labels_test = np.load(file_path+'Data/'+'labels_test.npy')
    print("hi")
    if my_dict["Train"]:
        my_model = CNN(12, 12, 6)
        my_model.weight_init(mean=0, std=0.02)

        my_train(data_train, labels_train, my_model, num_epochs=train_prameters.epoch, batch_size=train_prameters.batch, file_path, checkpoint_bool=True)





