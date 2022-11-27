import torch


def training_model(model, optimizer_fn, loss_fn, metric_fn,
                   data_train, data_test,
                   learning_rate=1e-3, current_device="cpu",
                   epochs=None, max_epochs=1e4, min_score=0.95,
                   valid_period=5, printed=False,
                   with_addition=False, new_optim=False):
    """Train model
    Parameters:
        model (torch.nn.Module object): neural network model
        optimizer_fn (type): optimizer type (e.g. optim.SGD) or optimizer itself
        loss_fn (function): loss function to measure model error
        metric_fn (function): function to evaluate model accuracy
        data_train (torch.utils.data.DataLoader): train dataloader
        data_test (torch.utils.data.DataLoader): test dataloader
        learning_rate (float): speed for optimization method
        current_device (str): current available device (cuda or cpu)
        epochs (int): the number of epochs (for training phase)
        max_epochs (int): the constraint for training epochs
        min_score (float): the constraint for score value
        valid_period (int): how often we evaluate our model (each 'valid_period' epoch)
        printed (bool): print score and loss in validation phase or not
        with_addition (bool): do we use loss function with adj(X) or not (see 'new_loss.png' file)
        new_optim (bool): do we use new optimizer or not
    """
    model = model.to(current_device)
    if type(optimizer_fn) == type:
        optimizer = optimizer_fn(params=model.parameters(),
                                 lr=learning_rate)
    else:
        optimizer = optimizer_fn

    loss_train_array = []
    loss_test_array = []
    score_test_array = []

    if epochs:
        for epoch in range(1, epochs+1):
            loss_epoch = train_epoch(model, optimizer, loss_fn, metric_fn, data_train, current_device, with_addition, new_optim)
            loss_train_array.append(loss_epoch)

            # evaluate
            if epoch % valid_period == 0:
                loss, metric = validation_epoch(model, loss_fn, metric_fn, data_test, current_device)
                loss_test_array.append(loss)
                score_test_array.append(metric)

                if printed:
                    print(f'epoch {epoch}: loss = {loss:.3f} and score = {metric:.3f}')
        return loss_train_array, loss_test_array, score_test_array

    else:
        metric = 0.0
        epoch = 0
        while (metric < min_score) and (epoch < max_epochs):
            # train
            loss_epoch = train_epoch(model, optimizer, loss_fn, metric_fn, data_train, current_device, with_addition, new_optim)
            loss_train_array.append(loss_epoch)

            # evaluate
            if epoch % valid_period == 0:
                loss, metric = validation_epoch(model, loss_fn, metric_fn, data_test, current_device)
                loss_test_array.append(loss)
                score_test_array.append(metric)
                if printed:
                    print(f'epoch {epoch+1}: loss = {loss:.3f} and score = {metric:.3f}')
            epoch += 1
        return epoch, loss_train_array, loss_test_array, score_test_array


def train_epoch(model, optimizer, loss_fn, metric_fn, data_train, current_device="cpu", with_addition=False, new_optim=False):
    """One epoch in train cycle"""
    mean_loss_epoch = 0.0
    model.train()
    for X_batch, y_batch in data_train:
        X_batch, y_batch = X_batch.to(current_device), y_batch.to(current_device)

        # forward pass
        predicted = model(X_batch)

        # loss computation
        if with_addition:
            # if we use adjoint matrix (functionality for new optimizer)
            determinant = torch.det(X_batch)
            inverse_batch = torch.linalg.inv(X_batch)
            adjoint = determinant * inverse_batch
            loss = loss_fn(adjoint @ predicted, adjoint @ y_batch)

        else:
            loss = loss_fn(predicted, y_batch)

        mean_loss_epoch += loss
        # zero gradient
        optimizer.zero_grad()

        # backpropagation (compute gradient)
        loss.backward()

        # update model parameters
        if new_optim:
            optimizer.step(determinant_X_batch=determinant)
        else:
            optimizer.step()

    mean_loss_epoch /= len(data_train)
    return mean_loss_epoch


def validation_epoch(model, loss_fn, metric_fn, data_test, current_device):
    with torch.inference_mode():
        metric, loss = 0.0, 0.0
        for X_batch, y_batch in data_test:
            X_batch, y_batch = X_batch.to(current_device), y_batch.to(current_device)
            predicted = model(X_batch)
            loss += loss_fn(predicted, y_batch)
            metric += metric_fn(predicted, y_batch)
        metric /= len(data_test)
        loss /= len(data_test)
    return loss, metric
