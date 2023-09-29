import numpy as np

class prameters_class():
    def __init__(self, M,N_q,SNR,snapshot,D,teta_range,monte,delta,Res,dictio):
        self.M = M
        self.N_q = N_q
        self.D = D
        self.teta_range = teta_range
        self.SNR = SNR
        self.snapshot = snapshot
        self.monte = monte
        self.delta = delta
        self.Res = Res
        self.dictio = dictio
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