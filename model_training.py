import torch
import torch.nn as nn
import numpy as np

def self_prediction_train(model, x, lr=0.01, num_epochs=20, batch=64,
                          loss_fn=nn.MSELoss(), shuffle=True, verbose=0):
    if not torch.is_tensor(x):
        x = torch.Tensor(x)

    no_samples = x.shape[0]
    no_iter_per_epoch = no_samples//batch
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        if shuffle:
            random_index = torch.randperm(no_samples)
            x = x[random_index]
            y = y[random_index]
        for j in range(no_iter_per_epoch):
            if j != no_iter_per_epoch - 1:
                out = model(x[batch*j:batch*(j+1)])
                loss = loss_fn(x[batch*j:batch*(j+1)], out)
            else:
                out = model(x[batch*j:])
                loss = loss_fn(x[batch*j:], out)

            # Calculate gradient
            loss.backward()

            # Update weights
            optimizer.step()

            # Reset gradient
            optimizer.zero_grad()

            if verbose == 1 and j == no_iter_per_epoch - 1:
                print(f"Epoch {epoch+1}/{num_epochs}: loss = {loss}")


def classifier_train(model, x, y, lr=0.01, num_epochs=20, batch=64,
                     loss_fn=nn.BCELoss(), shuffle=True, verbose=0):
    if not torch.is_tensor(x):
        x = torch.Tensor(x)
    if not torch.is_tensor(y):
        y = torch.Tensor(y)

    no_samples = x.shape[0]
    no_iter_per_epoch = no_samples//batch
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        if shuffle:
            random_index = torch.randperm(no_samples)
            x = x[random_index]
            y = y[random_index]
        for j in range(no_iter_per_epoch):
            if j != no_iter_per_epoch - 1:
                out = model(x[batch*j:batch*(j+1)])
                loss = loss_fn(out, y[batch*j:batch*(j+1)])
            else:
                out = model(x[batch*j:])
                loss = loss_fn(out, y[batch*j:])

            # Calculate gradient
            loss.backward()

            # Update weights
            optimizer.step()

            # Reset gradient
            optimizer.zero_grad()

            if verbose == 1 and j == no_iter_per_epoch - 1:
                out = model(x)
                out[out>0.5] = 1
                out[out<=0.5] = 0
                acc = (out == y).sum() / x.shape[0]
                print(f"Epoch {epoch+1}/{num_epochs}: loss = {loss.item(): 4f}, "
                      f"accuracy = {acc: 4f}")