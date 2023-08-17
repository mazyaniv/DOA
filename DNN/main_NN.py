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
    file_path = 'C:/Users/Yaniv/PycharmProjects/DOA/DNN/'  # '/home/mazya/DNN/'
    my_parameters = prameters_class(10, 0, 2, [0, 60], 0, 400, 10)
    train_prameters = train_prameters(1000, 50, 100, 10, 0.001)
    my_dict = {"Generate new data":True,
               "Train": True, "Test": True}

    if my_dict["Generate new data"]:
        generate_data(my_parameters,train_prameters,file_path)

    data_file_path = file_path+'Data/'
    my_data = My_data(data_file_path)

    if my_dict["Train"]:
        my_model = CNN(my_parameters.teta_range)
        my_model.weight_init(mean=0, std=0.02)
        my_train(my_data.data_train, my_data.labels_train, my_model,my_parameters.teta_range, num_epochs=train_prameters.epoch,
                 batch_size=train_prameters.batch, checkpoint_path='Trained_Model/',checkpoint_bool=True)

    elif my_dict["Test"]:
        Model = CNN(my_parameters.teta_range)
        Model.load_state_dict(torch.load(file_path+'Trained_Model/'+'model_checkpoint.pth'))
        Model.eval()
        print("RMSE:",test_model(Model,my_data.data_test,my_data.labels_test,my_parameters.C))
