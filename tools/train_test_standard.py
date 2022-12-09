"""
help functionality for train models with classic optimizers
"""

import torch


def standard_training(model,
                      optimizer_fn,
                      loss_fn,
                      metric_fn,
                      data_train,
                      data_test,
                      learning_rate=1e-3,
                      current_device="cpu",
                      epochs=None,
                      max_epochs=1e4,
                      min_score=0.95,
                      printed=False):
    """Train model with standard optimizers
    Parameters:
        model (torch.nn.Module object): neural network model
        optimizer_fn (type, torch.optim): optimizer type (e.g. optim.SGD) or optimizer itself
        loss_fn (function): loss function to measure model error
        metric_fn (function): function to evaluate model accuracy
        data_train (torch.utils.data.DataLoader): train dataloader
        data_test (torch.utils.data.DataLoader): test dataloader
        learning_rate (float): speed for optimization method
        current_device (str): current available device (cuda or cpu)
        epochs (int): the number of epochs (for training phase).
                      If 'None' training continues until the metric value reaches 'min_score'
        max_epochs (int): the constraint for training epochs (work if epochs is `None`)
        min_score (float): the constraint for score value (works if epochs is `None`)
        printed (bool): print score and loss in validation phase or not
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

    if epochs is not None:
        for epoch in range(1, epochs+1):
            mean_loss_epoch = standard_train_epoch(model, loss_fn, optimizer, data_train, current_device="cpu")
            loss_train_array.append(mean_loss_epoch)
            # evaluate
            mean_loss_test, mean_metric_test = standard_valid_epoch(model, loss_fn, metric_fn, data_test,
                                                                    current_device='cpu')
            loss_test_array.append(mean_loss_test)
            score_test_array.append(mean_metric_test)
            if printed:
                print(f'epoch {epoch}/{epochs}: loss = {mean_loss_test:.3f} and score = {mean_metric_test:.3f}')
        return loss_train_array, loss_test_array, score_test_array

    else:
        mean_metric_test = 0.0
        epoch = 1
        while (mean_metric_test < min_score) and (epoch <= max_epochs):
            # train
            mean_loss_epoch = standard_train_epoch(model, loss_fn, optimizer, data_train, current_device="cpu")
            loss_train_array.append(mean_loss_epoch)

            # evaluate
            mean_loss_test, mean_metric_test = standard_valid_epoch(model, loss_fn, metric_fn, data_test, current_device='cpu')
            loss_test_array.append(mean_loss_test)
            score_test_array.append(mean_metric_test)

            if printed:
                print(f'epoch {epoch}: loss = {mean_loss_test:.3f} and score = {mean_metric_test:.3f}')
            epoch += 1

        return epoch, loss_train_array, loss_test_array, score_test_array


def standard_train_epoch(model,
                         loss_fn,
                         optimizer,
                         data_train,
                         current_device="cpu"):
    """One epoch in train cycle when uses standard optimizers"""
    loss_epoch = 0.0
    model.train()
    for X_batch, y_batch in data_train:
        X_batch, y_batch = X_batch.to(current_device), y_batch.to(current_device)
        # forward pass
        predicted = model(X_batch)
        loss = loss_fn(predicted, y_batch)
        loss_epoch += loss.detach()

        # zero gradient
        optimizer.zero_grad()

        # backpropagation (compute gradient)
        loss.backward()

        # update model parameters
        optimizer.step()
    return loss_epoch / len(data_train)


def standard_valid_epoch(model, loss_fn, metric_fn, data_test, current_device='cpu'):
    """Validation phase in epoch"""
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
