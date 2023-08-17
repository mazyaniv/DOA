import torch.optim as optim
import numpy as np
import torch
from functions_NN import *

def my_train(train,labels,model, num_epochs, batch_size,file_path,checkpoint_bool=False):
  model.train()
  optimizer = optim.Adam(model.parameters())
  for epoch in range(num_epochs):
    if epoch>1:
      print(loss)
    if epoch%7 == 0:
      print(f"Epoch number {epoch}")
    #shuffle the data each epoch
    train_size = train.shape[1]
    per = np.random.permutation(train_size)
    train = train[:,per]
    labels = labels[per]

    for i in range(0, train_size, batch_size):
        if (i + batch_size) > train_size:
            break
        # get the input and targets of a minibatch
        z,s = get_batch(train, labels, i, i+batch_size)
        optimizer.zero_grad()
        loss = BCEWithLogitsLoss(model,z,s) # compute the total loss
        loss.backward()
        optimizer.step()

  checkpoint_path = file_path+'Trained_Model/'
  if checkpoint_bool:
        torch.save(model.state_dict(), checkpoint_path)
  print("Finish")