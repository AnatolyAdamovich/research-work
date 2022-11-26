import torch


def training_model(model, optimizer_fn, loss_fn, metric_fn,
                   data_train, data_test, input_features=10,
                   learning_rate=1e-3, current_device="cpu",
                   epochs=None, max_epochs=1e4, min_score=0.95,
                   valid_period=5, printed=False,
                   with_addition=False):
    """Train model
    Parameters:
        model (torch.nn.Module object): neural network model
        optimizer_fn (function): optimizer to update parameters
        loss_fn (function): loss function to measure model error
        metric_fn (function): function to evaluate model accuracy
        data_train (torch.utils.data.DataLoader): train dataloader
        data_test (torch.utils.data.DataLoader): test dataloader
        input_features (int): how many features our data has
        learning_rate (float): speed for optimization method
        current_device (str): current available device (cuda or cpu)
        epochs (int): the number of epochs (for training phase)
        max_epochs (int): the constraint for training epochs
        min_score (float): the constraint for score value
        valid_period (int): how often we evaluate our model (each 'valid_period' epoch)
        printed (bool): print score and loss in validation phase or not
        with_addition (bool): do we use loss function with adj(X) or not (see 'new_loss.png' file)
    """
    model = model.to(current_device)
    optimizer = optimizer_fn(params=model.parameters(),
                             lr=learning_rate)
    if epochs:
        for epoch in range(1, epochs+1):
            train_epoch(model, optimizer, loss_fn, metric_fn, data_train, current_device, with_addition)
            # evaluate
            if epoch % valid_period == 0:
                loss, metric = validation_epoch(model, loss_fn, metric_fn, data_test, current_device)
                if printed:
                    print(f'epoch {epoch}: loss = {loss:.3f} and score = {metric:.3f}')
    else:
        metric = 0.0
        epoch = 0
        while (metric < min_score) and (epoch < max_epochs):
            # train
            train_epoch(model, optimizer, loss_fn, metric_fn, data_train, current_device, with_addition)

            # evaluate
            if epoch % valid_period == 0:
                loss, metric = validation_epoch(model, loss_fn, metric_fn, data_test, current_device)
                if printed:
                    print(f'epoch {epoch}: loss = {loss:.3f} and score = {metric:.3f}')
            epoch += 1
        return epoch, metric


def train_epoch(model, optimizer, loss_fn, metric_fn, data_train, current_device, with_addition=False):
    """One epoch in train cycle"""

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

        # zero gradient
        optimizer.zero_grad()

        # backpropagation (compute gradient)
        loss.backward()

        # update model parameters
        optimizer.step()


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
