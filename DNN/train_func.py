def my_train(train,labels,P,model, num_epochs, batch_size, learning_rate, weight_decay=1e-9,
          checkpoint_bool=False):
  #define optimizer

  model.train()
  # pos_weight = torch.ones([M])  # All weights are equal to 1
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
  checkpoint_path = file_path+'SNR={}_snap={}_Q={}/'.format(SNR,snap,P)+'SNR='+str(SNR)+'_snap='+str(snap)+"_Q="+str(Q)
  if checkpoint_bool:
        torch.save(model.state_dict(), checkpoint_path)
  print("Finish")