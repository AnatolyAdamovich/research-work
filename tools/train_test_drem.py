"""
help functions for training cycle with new DREM - optimizer
"""
import torch
from .train_test_help_functions import validation_epoch


def drem_opt_training(model,
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
    """Train model with DREM optimizer
    Parameters:
        model (torch.nn.Module object): neural network model
        optimizer_fn (type): optimizer type (optim.DREMOptimizer) or optimizer itself
        loss_fn (function): loss function to measure model error
        metric_fn (function): function to evaluate model accuracy
        data_train (torch.utils.data.DataLoader): train dataloader
        data_test (torch.utils.data.DataLoader): test dataloader
        learning_rate (float): speed for optimization method
        current_device (str): current available device (cuda or cpu)
        epochs (int): the number of epochs (for training phase)
        max_epochs (int): the constraint for training epochs
        min_score (float): the constraint for score value
        printed (bool): print score and loss in validation phase or not
    """
    model = model.to(current_device)
    n_of_input_feature = len(next(iter(model.parameters()))[0])
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
            mean_loss_epoch = drem_train_epoch(model, optimizer, loss_fn,
                                               data_train, n_of_input_feature, current_device)
            loss_train_array.append(mean_loss_epoch)
            # evaluate
            mean_loss_test, mean_metric_test = validation_epoch(model, loss_fn, metric_fn, data_test, current_device)
            loss_test_array.append(mean_loss_test)
            score_test_array.append(mean_metric_test)

            if printed:
                print(f'epoch {epoch}/{epochs}: loss = {mean_loss_test:.3f} and score = {mean_metric_test:.3f}')
        return loss_train_array, loss_test_array, score_test_array

    else:
        mean_metric_test = 0.0
        epoch = 0
        while (mean_metric_test < min_score) and (epoch <= max_epochs):
            # train
            mean_loss_epoch = drem_train_epoch(model, optimizer, loss_fn, data_train,
                                               n_of_input_feature, current_device)
            loss_train_array.append(mean_loss_epoch)
            # evaluate
            mean_loss_test, mean_metric_test = validation_epoch(model, loss_fn, metric_fn, data_test, current_device)
            loss_test_array.append(mean_loss_test)
            score_test_array.append(mean_metric_test)

            if printed:
                print(f'epoch {epoch+1}: loss = {mean_loss_test:.3f} and score = {mean_metric_test:.3f}')
            epoch += 1

        return epoch, loss_train_array, loss_test_array, score_test_array


def drem_train_epoch(model,
                     optimizer,
                     loss_fn,
                     data_train,
                     n_feature=None,
                     current_device="cpu"):
    """One epoch in train cycle"""
    loss_epoch = 0.0
    model.train()
    for X_batch, y_batch in data_train:
        X_batch, y_batch = X_batch.to(current_device), y_batch.to(current_device)
        if n_feature is None:
            raise ValueError("`n_feature` should be integer when we use new optimizer")
        # forward pass
        predicted = model(X_batch)
        loss_estimation = loss_fn(predicted, y_batch).detach()
        loss_epoch += loss_estimation

        len_batch = len(X_batch)
        if len_batch == n_feature:
            # 1nd case: don't need to cut batch

            # loss computation
            determinant = torch.det(X_batch)
            inverse_batch = torch.linalg.inv(X_batch)
            adjoin = determinant * inverse_batch
            loss = loss_fn(adjoin @ predicted, adjoin @ y_batch)

            # zero gradients
            optimizer.zero_grad()

            # backpropagation (compute gradient)
            loss.backward()

            # parameters update
            optimizer.step(det_batch=determinant)
        else:
            # 2nd case: need to cut batch into `s` parts
            s = len_batch // n_feature

            for i in range(0, len_batch, n_feature):
                # loss computation
                determinant = torch.det(X_batch[i:(i + n_feature), :])
                inverse_batch = torch.linalg.inv(X_batch[i:(i + n_feature), :])
                adjoin = determinant * inverse_batch
                loss = loss_fn(adjoin @ predicted[i:(i + n_feature), :], adjoin @ y_batch[i:(i + n_feature), :])

                # zero gradients
                optimizer.zero_grad()

                # backpropagation (compute gradient)
                loss.backward(retain_graph=True)

                # parameters update
                optimizer.step(det_batch=determinant, partition=s)
                s -= 1

    return loss_epoch / len(data_train)