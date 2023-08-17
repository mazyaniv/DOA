import numpy as np

class prameters_class():
    def __init__(self, M,P,D,teta_range,SNR,snapshot):
        self.M = M
        self.P = P
        self.D = D
        self.teta_range = teta_range
        self.SNR = SNR
        self.snap = snapshot
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

if __name__ == "__main__":
    print("Not main file")