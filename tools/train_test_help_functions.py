"""general help functions for training & evaluate cycles"""

import torch
import matplotlib.pyplot as plt


def validation_epoch(model, loss_fn, metric_fn, data_test, current_device='cpu'):
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


def plot_results(loss_train, loss_test, score_test):
    """help function to visualize results"""
    plt.figure(figsize=(18, 7))
    plt.subplot(1, 2, 1)
    plt.plot(loss_train, c='r', label='train')
    plt.plot(loss_test, c='b', label='test')
    plt.title('Loss function', size=20)
    plt.xlabel('epochs', size=20)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(score_test)
    plt.title('Score on test set', size=20)
    plt.xlabel('epochs', size=20)



# def training_model(model, optimizer_fn, loss_fn, metric_fn,
#                    data_train, data_test,
#                    learning_rate=1e-3, current_device="cpu",
#                    epochs=None, max_epochs=1e4, min_score=0.95,
#                    valid_period=5, printed=False,
#                    with_addition=False, new_optim=False):
#     """Train model
#     Parameters:
#         model (torch.nn.Module object): neural network model
#         optimizer_fn (type): optimizer type (e.g. optim.SGD) or optimizer itself
#         loss_fn (function): loss function to measure model error
#         metric_fn (function): function to evaluate model accuracy
#         data_train (torch.utils.data.DataLoader): train dataloader
#         data_test (torch.utils.data.DataLoader): test dataloader
#         learning_rate (float): speed for optimization method
#         current_device (str): current available device (cuda or cpu)
#         epochs (int): the number of epochs (for training phase)
#         max_epochs (int): the constraint for training epochs
#         min_score (float): the constraint for score value
#         valid_period (int): how often we evaluate our model (each 'valid_period' epoch)
#         printed (bool): print score and loss in validation phase or not
#         with_addition (bool): do we use loss function with adj(X) or not (see 'new_loss.png' file)
#         new_optim (bool): do we use new optimizer or not
#     """
#     model = model.to(current_device)
#     n_of_input_feature = len(next(iter(model.parameters()))[0])
#     if type(optimizer_fn) == type:
#         optimizer = optimizer_fn(params=model.parameters(),
#                                  lr=learning_rate)
#     else:
#         optimizer = optimizer_fn
#
#     loss_train_array = []
#     loss_test_array = []
#     score_test_array = []
#
#     if epochs:
#         for epoch in range(1, epochs+1):
#             loss_epoch = train_epoch(model, optimizer, loss_fn, metric_fn, data_train,
#                                      n_of_input_feature, current_device, with_addition, new_optim)
#             loss_train_array.append(loss_epoch)
#             # evaluate
#             loss, metric = validation_epoch(model, loss_fn, metric_fn, data_test, current_device)
#             loss_test_array.append(loss)
#             score_test_array.append(metric)
#             if (epoch % valid_period == 0) and printed:
#                 print(f'epoch {epoch}/{epochs}: loss = {loss:.3f} and score = {metric:.3f}')
#         return loss_train_array, loss_test_array, score_test_array
#
#     else:
#         metric = 0.0
#         epoch = 1
#         while (metric < min_score) and (epoch <= max_epochs):
#             # train
#             loss_epoch = train_epoch(model, optimizer, loss_fn, metric_fn, data_train,
#                                      n_of_input_feature, current_device, with_addition, new_optim)
#             loss_train_array.append(loss_epoch)
#
#             # evaluate
#             loss, metric = validation_epoch(model, loss_fn, metric_fn, data_test, current_device)
#             loss_test_array.append(loss)
#             score_test_array.append(metric)
#
#             if (epoch % valid_period == 0) and printed:
#                 print(f'epoch {epoch}: loss = {loss:.3f} and score = {metric:.3f}')
#             epoch += 1
#
#         return epoch, loss_train_array, loss_test_array, score_test_array
#
#
# def train_epoch(model, optimizer, loss_fn, metric_fn, data_train, n_feature=None,
#                 current_device="cpu", with_addition=False, new_optim=False):
#     """One epoch in train cycle"""
#     mean_loss_epoch = 0.0
#     model.train()
#     for X_batch, y_batch in data_train:
#         X_batch, y_batch = X_batch.to(current_device), y_batch.to(current_device)
#
#         if new_optim:
#             mean_loss_epoch = _new_optim_phase(model, optimizer, loss_fn, X_batch, y_batch, n_feature, mean_loss_epoch)
#         else:
#             mean_loss_epoch = _classic_phase(model, optimizer, loss_fn, X_batch, y_batch, mean_loss_epoch, with_addition)
#
#     mean_loss_epoch /= len(data_train)
#     return mean_loss_epoch


# def _new_optim_phase(model, optimizer, loss_fn, X_batch, y_batch, n_feature, mean_loss_epoch):
#     if n_feature is None:
#         raise ValueError("`n_feature` should be integer when we use new optimizer")
#
#     # forward pass
#     predicted = model(X_batch)
#     loss2 = loss_fn(predicted, y_batch).detach()
#     mean_loss_epoch += loss2
#
#     len_batch = len(X_batch)
#     if len_batch == n_feature:
#         # loss computation
#         determinant = torch.det(X_batch)
#         inverse_batch = torch.linalg.inv(X_batch)
#         adjoin = determinant * inverse_batch
#         loss1 = loss_fn(adjoin @ predicted, adjoin @ y_batch)
#
#         # zero gradients
#         optimizer.zero_grad()
#
#         # backpropagation (compute gradient)
#         loss1.backward()
#
#         # parameters update
#         optimizer.step(det_batch=determinant)
#     else:
#         s = len_batch // n_feature
#         for i in range(0, len_batch, n_feature):
#
#             # loss computation
#             determinant = torch.det(X_batch[i:(i+n_feature), :])
#             inverse_batch = torch.linalg.inv(X_batch[i:(i+n_feature), :])
#             adjoin = determinant * inverse_batch
#             loss1 = loss_fn(adjoin @ predicted[i:(i+n_feature), :], adjoin @ y_batch[i:(i+n_feature), :])
#
#             # zero gradients
#             optimizer.zero_grad()
#
#             # backpropagation (compute gradient)
#             loss1.backward(retain_graph=True)
#
#             # parameters update
#             optimizer.step(det_batch=determinant, partition=s)
#             s -= 1
#
#     return mean_loss_epoch
#
#
# def _classic_phase(model, optimizer, loss_fn, X_batch, y_batch, mean_loss_epoch, with_addition):
#     # forward pass
#     predicted = model(X_batch)
#     loss2 = loss_fn(predicted, y_batch).detach()
#     mean_loss_epoch += loss2
#
#     # loss computation
#     if with_addition:
#         # use adjoin matrix
#         # (functionality for new optimizer, but can also be used with classic optimizers)
#         determinant = torch.det(X_batch)
#         inverse_batch = torch.linalg.inv(X_batch)
#         adjoin = determinant * inverse_batch
#         loss1 = loss_fn(adjoin @ predicted, adjoin @ y_batch)
#     else:
#         loss1 = loss_fn(predicted, y_batch)
#
#     # zero gradient
#     optimizer.zero_grad()
#
#     # backpropagation (compute gradient)
#     loss1.backward()
#
#     # update model parameters
#     optimizer.step()
#
#     return mean_loss_epoch
