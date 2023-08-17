import torch
import numpy as np
import math
def Matrix(teta):
    global M, D
    teta = np.radians(teta)
    B = np.zeros((M, D))
    for j in range(D):
        B[:, j] = np.exp(-1j * np.pi * np.arange(M) * np.sin(teta[j]))
    return B

def observ(teta, SNR, snap):
    A = Matrix(teta)
    real_s = np.random.normal(0, 1 / math.sqrt(2), (D, snap))
    im_s = np.random.normal(0, 1 / math.sqrt(2), (D, snap))
    s = real_s + 1j * im_s
    s_samp = s.reshape(D, snap)

    real_n = np.random.normal(0, (10 ** (-SNR / 20)) / math.sqrt(2), (M, snap))
    im_n = np.random.normal(0, (10 ** (-SNR / 20)) / math.sqrt(2), (M, snap))
    n = real_n + 1j * im_n
    n_samp = n.reshape(M, snap)
    x_a_samp = (A @ s_samp) + n_samp
    return x_a_samp

def quantize_part(A,P):
        mask = np.zeros(np.shape(A),dtype=complex)
        mask[:P,:] = (1/math.sqrt(2))*(np.sign(A[:P,:].real-(thresh_real))+(1j*(np.sign(A[:P,:].imag-((thresh_im))))))
        mask[P:,:] = A[P:,:]
        return mask

def get_batch(R, labels, inx_min, inx_max):
  xt = R[:,inx_min:inx_max]
  st = labels[inx_min:inx_max]
  permutation = np.random.permutation(inx_max - inx_min)
  return xt[:,permutation] , make_onehot(st[permutation]) #Shift or Onehot

def make_onehot(target): #batch
    OneHotMask = np.zeros((target.shape[0],teta_max-teta_min+1))
    for i in range(target.shape[0]):
      for j in range(target.shape[1]):
        OneHotMask[i,int(target[i,j]+shift)] = 1 #Shift
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
