


# Defining Loss
criterion = nn.MSELoss()

#Defining Hyperparemeters
no_epochs = 100 #very low, but computational power not sufficient for more iterations
batch = 1024
learning_rate = 0.001

#Using the Adam Method for Stochastic Optimisation
#optimiser = optim.Adam(model.parameters(), lr=learning_rate)

galaxy_types = ['LRG', 'ELG', 'QSO']

for gal in galaxy_types:
    model = Net().to(device)
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    print("GALAXY TYPE: ", gal)
    print()
    traindata = DensitySurvey(train_df_geo, gal)
    scaler_in, scaler_out = traindata.__getscaler__()
    testdata = DensitySurvey(test_df_geo, gal, scaler_in, scaler_out)

    time_start = time.time()

    for epoch in range(no_epochs):
        loss_per_epoch = 0

        #loading the training data from trainset and shuffling for each epoch
        trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch, shuffle = True)

        for i, batch_no in enumerate(trainloader, 0):

            #Put Model into train mode
            model.train()

            #Extract inputs and associated labels from dataloader batch
            inputs = batch_no[0].to(device)
            labels = batch_no[1].to(device)

            #Zero-out the gradients before backward pass (pytorch stores the gradients)
            optimiser.zero_grad()

            #Predict outputs (forward pass)
            predictions =  model(inputs)

            #Compute Loss
            loss = criterion(predictions, labels)

            #Backpropagation
            loss.backward()

            #Perform one step of gradient descent
            optimiser.step()

            #Append loss to the general loss for this one epoch
            loss_per_epoch += loss.item()
        if epoch % 10 == 0:
            print("Loss for Epoch", epoch, ": ", loss_per_epoch)

    time_end = time.time()
    time_passed = time_end - time_start
    print()
    print(f"{time_passed/60:.5} minutes ({time_passed:.3} seconds) taken to train the model")


    model.eval()
    y_pred = np.array([])
    testloader = torch.utils.data.DataLoader(testdata, batch_size=batch, shuffle=False)


    for batch_no in testloader:

        #Split dataloader
        inputs = batch_no[0].to(device)
        labels = batch_no[1].to(device)

        #Forward pass through the trained network
        outputs = model(inputs)

        #Get predictions and append to label array + count number of correct and total
        y_pred = np.append(y_pred, outputs.detach().numpy())

    y_gold = testdata.target


    print()
    print(f"Neural Net R^2 for {gal}, Geometric :  {metrics.r2_score(y_gold, y_pred)}.")
    print(f"Neural Net MSE for {gal}, Geometric :  {metrics.mean_squared_error(y_gold, y_pred)}.")
