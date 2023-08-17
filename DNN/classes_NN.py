import numpy as np

class prameters_class():
    def __init__(self, M,N_q,SNR,snapshot,teta_range,D,C):
        self.M = M
        self.N_q = N_q
        self.SNR = SNR
        self.snap = snapshot
        self.teta_range = teta_range
        self.D = D
        self.C = C
class train_prameters():
    def __init__(self, N,test_size,batch,epoch,learning_rate, weight_decay=1e-9):
        self.N = N
        self.test_size = test_size
        self.batch = batch
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
class Matrix_class():
    def __init__(self, M, teta):
        self.teta = teta
        self.M = M
        self.D = len(teta)
        self.A = np.zeros((self.M, self.D), dtype=complex)
    def matrix(self):
        teta = np.radians(self.teta)
        A_mask = np.zeros((self.M, self.D), dtype=complex)
        for j in range(self.D):
            A_mask[:, j] = np.exp(-1j * np.pi * np.arange(self.M) * np.sin(teta[j]))
        self.A = A_mask
        return self.A

class My_data():
    def __init__(self, file_path,my_parameters):
        self.data_train = np.load(file_path+f'data_train_N_a={my_parameters.M-my_parameters.N_q}_N_q={my_parameters.N_q}_SNR={my_parameters.SNR}.npy')
        self.labels_train = np.load(file_path+f'labels_train_N_a={my_parameters.M-my_parameters.N_q}_N_q={my_parameters.N_q}_SNR={my_parameters.SNR}.npy')
        self.data_test = np.load(file_path+f'data_test_N_a={my_parameters.M-my_parameters.N_q}_N_q={my_parameters.N_q}_SNR={my_parameters.SNR}.npy')
        self.labels_test = np.load(file_path+f'labels_test_N_a={my_parameters.M-my_parameters.N_q}_N_q={my_parameters.N_q}_SNR={my_parameters.SNR}.npy')

if __name__ == "__main__":
    print("Not main file")