import torch
import torch.nn as nn
import math
from classes_NN import *

def generate_data(my_parameters,train_prameters,file_path):
    data = np.zeros((my_parameters.D, train_prameters.N, my_parameters.M, my_parameters.M), dtype=np.complex128)
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
        Observ = quantize_part(observ(teta, my_parameters.M, my_parameters.SNR, my_parameters.snap),
                               my_parameters.N_q)  # Quantize
        R = np.cov(Observ)
        data[0, i, :, :] = np.triu(R, k=1).real
        data[1, i, :, :] = np.triu(R, k=1).imag

    data_train = data[:, :-train_prameters.test_size, :, :]
    labels_train = labels[:-train_prameters.test_size, :]
    data_test = data[:, -train_prameters.test_size:, :, :]
    labels_test = labels[-train_prameters.test_size:, :]

    np.save(file_path + 'Data/' + f'data_train_N_a={my_parameters.M-my_parameters.N_q}_N_q={my_parameters.N_q}_SNR={my_parameters.SNR}.npy', data_train)
    np.save(file_path + 'Data/' + f'labels_train_N_a={my_parameters.M-my_parameters.N_q}_N_q={my_parameters.N_q}_SNR={my_parameters.SNR}.npy', labels_train)
    np.save(file_path + 'Data/' + f'data_test_N_a={my_parameters.M-my_parameters.N_q}_N_q={my_parameters.N_q}_SNR={my_parameters.SNR}.npy', data_test)
    np.save(file_path + 'Data/' + f'labels_test_N_a={my_parameters.M-my_parameters.N_q}_N_q={my_parameters.N_q}_SNR={my_parameters.SNR}.npy', labels_test)
def observ(teta,M,SNR,snap):
    A = Matrix_class(M, teta).matrix()
    M = A.shape[0]
    D = A.shape[1]

    real_s = np.random.normal(0, 1 / math.sqrt(2), (D, snap))
    im_s = np.random.normal(0, 1 / math.sqrt(2), (D, snap))
    s = real_s + 1j * im_s

    s_samp = s.reshape(D, snap)
    real_n = np.random.normal(0, math.sqrt((10 ** (-SNR / 10))) / math.sqrt(2), (M, snap))
    im_n = np.random.normal(0, math.sqrt((10 ** (-SNR / 10))) / math.sqrt(2), (M, snap))
    n = real_n + 1j * im_n
    n_samp = n.reshape(M, snap)
    x_a_samp = (A @ s_samp) + n_samp
    return x_a_samp




def quantize_part(A,P,thresh=0):
        mask = np.zeros(np.shape(A),dtype=complex)
        mask[:P,:] = (1/math.sqrt(2))*(np.sign(A[:P,:].real-(thresh))+(1j*(np.sign(A[:P,:].imag-((thresh))))))
        mask[P:,:] = A[P:,:]
        return mask

def get_batch(R, labels, inx_min, inx_max,teta_range):
  xt = R[:,inx_min:inx_max]
  st = labels[inx_min:inx_max]
  permutation = np.random.permutation(inx_max - inx_min)
  return xt[:,permutation] , make_onehot(st[permutation],teta_range) #Shift or Onehot

def make_onehot(target,teta_range): #batch
    OneHotMask = np.zeros((target.shape[0],teta_range[1]-teta_range[0]+1))
    for i in range(target.shape[0]):
      for j in range(target.shape[1]):
        OneHotMask[i,int(target[i,j]+0)] = 1 #Shift=0
    return OneHotMask

# criterion = nn.MSELoss(reduction='sum')/torch.nn.CrossEntropyLoss()/torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) pos_weight = torch.ones([M])  # All weights are equal to 1
def CrossEntropyLoss(model,tensor1, tensor2):
    z = torch.tensor(tensor1, requires_grad=True,dtype=torch.float32).transpose(0, 1)
    z = model(z)
    s = torch.tensor(tensor2.squeeze(), requires_grad=True,dtype=torch.float32).long()
    return torch.nn.CrossEntropyLoss()(z,s)

def BCEWithLogitsLoss(model,tensor1, tensor2): #get batch with Onehot
    z = torch.tensor(tensor1, requires_grad=True,dtype=torch.float32).transpose(0, 1)
    z = model(z)
    s = torch.tensor(tensor2.squeeze(), requires_grad=True,dtype=torch.float32)
    return torch.nn.BCEWithLogitsLoss()(z,s)

####TODO####
def compute_rmse(model,tensor1, tensor2):
    z = torch.tensor(tensor1, requires_grad=True,dtype=torch.float32).transpose(0, 1)
    z = model(z)
    z = z.detach().numpy()
    z = np.argsort(z,1)[:,::-1]
    z = np.radians(z[:,:D].squeeze()) #sort if D>1
    z = torch.tensor(z.copy().squeeze(), requires_grad=True,dtype=torch.float32)
    s = torch.Tensor(np.radians(tensor2)).squeeze()
    mse = torch.mean((z - s)**2)
    return torch.sqrt(mse)

def MSE_loss(model,tensor1, tensor2):
    z = torch.tensor(tensor1, requires_grad=True,dtype=torch.float32).transpose(0, 1)
    z = model(z)
    s = torch.tensor(tensor2.squeeze(), requires_grad=True,dtype=torch.float32)
    return nn.MSELoss()(z,s)
# z,s = get_batch(data_train_vec[0], labels_train_vec[0], 0, 2)
# print(BCEWithLogitsLoss(CNN(12,12,6),z,s))

def test_model(model, data, labels,C):
    # print("Q=",Q)
    labels = labels.squeeze()
    model.eval()
    n = data.shape[1]
    z = torch.tensor(data, dtype=torch.float32).transpose(0, 1)
    with torch.no_grad():
        z = model(z)
        z = np.argsort(z.detach().numpy(), 1)[:, ::-1]
        z = z[:, :labels.shape[1]].squeeze()
        pred = np.sort(z, 1)[:, ::-1].squeeze()

        equal_elements = np.sum(np.all(pred == labels, axis=1))
        accuracy_percentage = equal_elements / n * 100.0

        sub_vec_old = pred - labels
        mask = np.logical_and(-C < np.min(sub_vec_old, axis=1), np.max(sub_vec_old, axis=1) < C)
        sub_vec_new = sub_vec_old[mask]

        RMSE = (np.sum(np.sum(np.power(sub_vec_new, 2), 1)) / (sub_vec_new.shape[0] * (pred.shape[1]))) ** 0.5
        # print(f"Accuracy: {accuracy_percentage:.2f}%")
        # print(f"RMSE : {RMSE:.2f}_Degrees,", "Number of relevant tests:",np.shape(sub_vec_new)[0])
        # print("======")
        return RMSE