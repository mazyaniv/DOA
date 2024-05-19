import numpy as np

class prameters_class():
    def __init__(self, M,N_q,snapshot,teta_range,D,C):
        self.M = M
        self.N_q = N_q
        self.snap = snapshot
        self.teta_range = teta_range
        self.D = D
        self.C = C
class train_prameters():
    def __init__(self, J,test_size,batch,epoch,learning_rate, weight_decay=1e-9):
        self.J = J
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
    def __init__(self, file_path):
        self.data_train = np.load(file_path+f'data_train.npy')
        self.labels_train = np.load(file_path+f'labels_train.npy')
        self.data_test = np.load(file_path+f'data_test.npy')
        self.labels_test = np.load(file_path+f'labels_test.npy')

if __name__ == "__main__":
    print("Not main file")