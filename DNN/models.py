import torch.nn as nn
import torch
import numpy as np
class CNN(nn.Module):
    def __init__(self,n1=12,n2=12,n3=6, kernel_size=3,padding_size=2,a=0.5):
        super(CNN, self).__init__()
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.activation = nn.ReLU()
        self.a = a
        self.sigmo = nn.Sigmoid()
        #nn.Dropout2d
        #nn.BatchNorm2d

        #convolution layers
        self.conv1=nn.Conv2d(in_channels=2, out_channels=self.n1, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv2=nn.Conv2d(in_channels=self.n1, out_channels=self.n2, kernel_size=self.kernel_size)
        self.conv3=nn.Conv2d(in_channels=self.n2, out_channels=self.n3, kernel_size=self.kernel_size)

        self.active = torch.nn.LeakyReLU(self.a)
        self.max = nn.MaxPool2d(2)
        self.drop = nn.Dropout(p=0.5, inplace=False)
        #fully-connected layers
        self.fc1 = nn.Linear(self.n3*16,1500)
        self.fc2 = nn.Linear(1500,1500)
        self.fc3 = nn.Linear(1500,61) #Resolution

    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

    def forward(self,x):
      #1st cnn layer
      x = self.conv1(x)
      x = self.drop(x)
      x = self.active(x)
      x = self.conv2(x)
      x = self.drop(x)
      x = self.active(x)
      x = self.drop(x)
      x = self.conv3(x)
      x = self.active(x)
      x = self.max(x) #torch.Size([n3, 4, 4])
      x = x.reshape(-1,self.n3*16)
      x = self.fc1(x)
      x = self.fc2(x)
      x = self.fc3(x)
      #x = self.sigmo(x)
      return x #torch.argmax(x,dim=1)

if __name__ == "__main__":
    file_path = '/home/mazya/DNN/Data/' #'C:/Users/Yaniv/PycharmProjects/DOA/DNN/'
    data_train = np.load(file_path + 'data_train.npy')
    x = data_train
    x = torch.tensor(x, requires_grad=True,dtype=torch.float32).transpose(0, 1)
    x = x[0:4]
    z = CNN()(x)
    print(z.shape)