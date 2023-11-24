import torch
import torch.nn as nn
import numpy as np

class EarlyStopping:
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.best_params = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_params = model.state_dict()
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.best_params = model.state_dict()


def self_prediction_train(model, x_train, lr=0.01, num_epochs=20, batch=64,
                          loss_fn=nn.MSELoss(), shuffle=True, verbose=0):
    """
    Function to train MAEEG model by self-prediction without validation.
    """

    if not torch.is_tensor(x_train):
        x_train = torch.Tensor(x_train)

    no_samples = x_train.shape[0]
    no_iter_per_epoch = no_samples//batch
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train_loss_list = []
        if shuffle:
            random_index = torch.randperm(no_samples)
            x_train = x_train[random_index]
        for j in range(no_iter_per_epoch):
            if j != no_iter_per_epoch - 1:
                out = model(x_train[batch*j:batch*(j+1)])
                loss = loss_fn(out, x_train[batch*j:batch*(j+1)])
            else:
                out = model(x_train[batch*j:])
                loss = loss_fn(out, x_train[batch*j:])

            train_loss_list.append(loss)

            # Calculate gradient
            loss.backward()

            # Update weights
            optimizer.step()

            # Reset gradient
            optimizer.zero_grad()

            if verbose == 1 and j == no_iter_per_epoch - 1:
                train_loss = sum(train_loss_list) / len(train_loss_list)
                print(f"Epoch {epoch+1}/{num_epochs}: train_loss = {train_loss}")


def self_prediction_train_valid(model, x_train, x_valid, lr=0.01, num_epochs=20,
                                patience=5, delta=0.01, batch=64, loss_fn=nn.MSELoss(),
                                shuffle=True, verbose=0):
    """
    Function to train MAEEG model by self-prediction with validation.
    """

    if not torch.is_tensor(x_train):
        x_train = torch.Tensor(x_train)

    if not torch.is_tensor(x_valid):
        x_train = torch.Tensor(x_valid)

    no_samples = x_train.shape[0]
    no_iter_per_epoch = no_samples//batch
    no_valid_samples = x_valid.shape[0]
    no_valid_loops = no_valid_samples//batch
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience, delta=delta)

    for epoch in range(num_epochs):
        train_loss_list = []
        if shuffle:
            random_index = torch.randperm(no_samples)
            x_train = x_train[random_index]
        for j in range(no_iter_per_epoch):
            if j != no_iter_per_epoch - 1:
                out = model(x_train[batch*j:batch*(j+1)])
                loss = loss_fn(out, x_train[batch*j:batch*(j+1)])
            else:
                out = model(x_train[batch*j:])
                loss = loss_fn(out, x_train[batch*j:])

            train_loss_list.append(loss)

            # Calculate gradient
            loss.backward()

            # Update weights
            optimizer.step()

            # Reset gradient
            optimizer.zero_grad()

            if j == no_iter_per_epoch - 1:
                # Validation
                valid_loss = 0.0
                for k in range(no_valid_loops):
                    if k != no_valid_loops - 1:
                        valid_out = model(x_valid[batch * k:batch * (k + 1)])
                        valid_loss = valid_loss + loss_fn(valid_out, x_valid[batch * k:batch * (k + 1)])
                    else:
                        valid_out = model(x_valid[batch * k:])
                        valid_loss = valid_loss + loss_fn(valid_out, x_valid[batch * k:])
                valid_loss = valid_loss / no_valid_loops
                early_stopping(val_loss=valid_loss, model=model)

                if verbose == 1:
                    train_loss = sum(train_loss_list)/len(train_loss_list)
                    print(f"Epoch {epoch+1}/{num_epochs}: train_loss = {train_loss}, valid_loss = {valid_loss}")

        if early_stopping.early_stop:
            model.load_state_dict(early_stopping.best_params)
            print("Early stopping.")
            break


def self_prediction_train_ver2(model, x_train, lr=0.01, freq_min=51,
                               freq_max=100, num_epochs=20, batch=64,
                               loss_fn=nn.MSELoss(), shuffle=True, verbose=0):
    """
    Function to train MAEEG model by self-prediction without validation, filtering out frequencies.
    """

    if not torch.is_tensor(x_train):
        x_train = torch.Tensor(x_train)

    no_samples = x_train.shape[0]
    no_iter_per_epoch = no_samples//batch
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train_loss_list = []
        if shuffle:
            random_index = torch.randperm(no_samples)
            x_train = x_train[random_index]
        for j in range(no_iter_per_epoch):
            if j != no_iter_per_epoch - 1:
                out = model(x_train[batch*j:batch*(j+1)])
                x_ifft = torch.Tensor(remove_frequency_component(x_train[batch*j:batch*(j+1)], freq_min=freq_min, freq_max=freq_max))
                loss = loss_fn(out, x_ifft)
            else:
                out = model(x_train[batch*j:])
                x_ifft = torch.Tensor(remove_frequency_component(x_train[batch*j:], freq_min=freq_min, freq_max=freq_max))
                loss = loss_fn(out, x_ifft)

            train_loss_list.append(loss)

            # Calculate gradient
            loss.backward()

            # Update weights
            optimizer.step()

            # Reset gradient
            optimizer.zero_grad()

            if verbose == 1 and j == no_iter_per_epoch - 1:
                train_loss = sum(train_loss_list) / len(train_loss_list)
                print(f"Epoch {epoch+1}/{num_epochs}: train_loss = {train_loss}")


def self_prediction_train_valid_ver2(model, x_train, x_valid, lr=0.01, freq_min=51,
                                     freq_max=100, num_epochs=20, batch=64, patience=5,
                                     delta=0.01, loss_fn=nn.MSELoss(), shuffle=True, verbose=0):
    """
    Function to train MAEEG model by self-prediction with validation, filtering out frequencies.
    """

    if not torch.is_tensor(x_train):
        x_train = torch.Tensor(x_train)

    if not torch.is_tensor(x_valid):
        x_valid = torch.Tensor(x_valid)

    no_samples = x_train.shape[0]
    no_iter_per_epoch = no_samples//batch
    no_valid_samples = x_valid.shape[0]
    no_valid_loops = no_valid_samples // batch
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience, delta=delta)

    for epoch in range(num_epochs):
        train_loss_list = []
        if shuffle:
            random_index = torch.randperm(no_samples)
            x_train = x_train[random_index]
        for j in range(no_iter_per_epoch):
            if j != no_iter_per_epoch - 1:
                out = model(x_train[batch*j:batch*(j+1)])
                x_ifft = torch.Tensor(remove_frequency_component(x_train[batch*j:batch*(j+1)], freq_min=freq_min, freq_max=freq_max))
                loss = loss_fn(out, x_ifft)
            else:
                out = model(x_train[batch*j:])
                x_ifft = torch.Tensor(remove_frequency_component(x_train[batch*j:], freq_min=freq_min, freq_max=freq_max))
                loss = loss_fn(out, x_ifft)

            train_loss_list.append(loss)

            # Calculate gradient
            loss.backward()

            # Update weights
            optimizer.step()

            # Reset gradient
            optimizer.zero_grad()

            if j == no_iter_per_epoch - 1:
                # Validation
                valid_loss = 0.0
                for k in range(no_valid_loops):
                    if k != no_valid_loops - 1:
                        valid_out = model(x_valid[batch * k:batch * (k + 1)])
                        valid_ifft = torch.Tensor(
                            remove_frequency_component(x_valid[batch*k:batch*(k+1)], freq_min=freq_min,
                                                       freq_max=freq_max)
                        )
                        valid_loss = valid_loss + loss_fn(valid_out, valid_ifft)
                    else:
                        valid_out = model(x_valid[batch * k:])
                        valid_ifft = torch.Tensor(
                            remove_frequency_component(x_valid[batch * k:], freq_min=freq_min,
                                                       freq_max=freq_max)
                        )
                        valid_loss = valid_loss + loss_fn(valid_out, valid_ifft)
                valid_loss = valid_loss / no_valid_loops
                early_stopping(val_loss=valid_loss, model=model)

                if verbose == 1:
                    train_loss = sum(train_loss_list) / len(train_loss_list)
                    print(f"Epoch {epoch+1}/{num_epochs}: train_loss = {train_loss}, valid_loss = {valid_loss}")

        if early_stopping.early_stop:
            model.load_state_dict(early_stopping.best_params)
            print("Early stopping.")
            break


def self_prediction_mask_and_frequency(model, x_train, x_valid, lr=0.01, freq_min=51, lamb=0.5,
                                       freq_max=100, num_epochs=20, batch=64, patience=5,
                                       delta=0.01, loss_fn=nn.MSELoss(), shuffle=True, verbose=0):
    """
    Function to train MAEEG model by self-prediction with validation, filtering out frequencies.
    """

    if not torch.is_tensor(x_train):
        x_train = torch.Tensor(x_train)

    if not torch.is_tensor(x_valid):
        x_valid = torch.Tensor(x_valid)

    no_samples = x_train.shape[0]
    no_iter_per_epoch = no_samples//batch
    no_valid_samples = x_valid.shape[0]
    no_valid_loops = no_valid_samples // batch
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience, delta=delta)

    for epoch in range(num_epochs):
        train_loss_list = []
        if shuffle:
            random_index = torch.randperm(no_samples)
            x_train = x_train[random_index]
        for j in range(no_iter_per_epoch):
            if j != no_iter_per_epoch - 1:
                out_filter = model(x_train[batch*j:batch*(j+1)])
                x_ifft = torch.Tensor(remove_frequency_component(x_train[batch*j:batch*(j+1)], freq_min=freq_min, freq_max=freq_max))

                out_mask = model.forward_mask(x_train[batch*j:batch*(j+1)])
                x_origin = x_train[batch*j:batch*(j+1)]

                loss_filter = loss_fn(out_filter, x_ifft)
                loss_mask = loss_fn(out_mask, x_origin)
                loss = lamb*loss_mask + (1-lamb)*loss_filter
            else:
                out_filter = model(x_train[batch*j:])
                x_ifft = torch.Tensor(remove_frequency_component(x_train[batch*j:], freq_min=freq_min, freq_max=freq_max))

                out_mask = model.forward_mask(x_train[batch*j:])
                x_origin = x_train[batch*j:]

                loss_filter = loss_fn(out_filter, x_ifft)
                loss_mask = loss_fn(out_mask, x_origin)
                loss = lamb*loss_mask + (1-lamb)*loss_filter

            train_loss_list.append(loss)

            # Calculate gradient
            loss.backward()

            # Update weights
            optimizer.step()

            # Reset gradient
            optimizer.zero_grad()

            if j == no_iter_per_epoch - 1:
                # Validation
                valid_loss = 0.0
                for k in range(no_valid_loops):
                    if k != no_valid_loops - 1:
                        valid_out_filter = model(x_valid[batch * k: batch * (k + 1)])
                        valid_ifft = torch.Tensor(
                            remove_frequency_component(x_valid[batch*k:batch*(k+1)], freq_min=freq_min,
                                                       freq_max=freq_max)
                        )

                        valid_out_mask = model.forward_mask(x_valid[batch * k: batch * (k + 1)])
                        valid_origin = x_valid[batch * k: batch * (k + 1)]

                        valid_loss = valid_loss + lamb * loss_fn(valid_out_mask, valid_origin) + (1-lamb) * loss_fn(valid_out_filter, valid_ifft)
                    else:
                        valid_out_filter = model(x_valid[batch * k:])
                        valid_ifft = torch.Tensor(
                            remove_frequency_component(x_valid[batch * k:], freq_min=freq_min,
                                                       freq_max=freq_max)
                        )

                        valid_out_mask = model.forward_mask(x_valid[batch * k:])
                        valid_origin = x_valid[batch * k:]

                        valid_loss = valid_loss + lamb * loss_fn(valid_out_mask, valid_origin) + (1-lamb) * loss_fn(valid_out_filter, valid_ifft)
                valid_loss = valid_loss / no_valid_loops
                early_stopping(val_loss=valid_loss, model=model)

                if verbose == 1:
                    train_loss = sum(train_loss_list) / len(train_loss_list)
                    print(f"Epoch {epoch+1}/{num_epochs}: train_loss = {train_loss}, valid_loss = {valid_loss}")

        if early_stopping.early_stop:
            model.load_state_dict(early_stopping.best_params)
            print("Early stopping.")
            break


def classifier_train(model, x_train, y_train, lr=0.01, num_epochs=20, batch=64,
                     loss_fn=nn.BCELoss(), shuffle=True, verbose=0):
    """
    Function to train classifier.
    """

    if not torch.is_tensor(x_train):
        x_train = torch.Tensor(x_train)
    if not torch.is_tensor(y_train):
        y_train = torch.Tensor(y_train)

    no_samples = x_train.shape[0]
    no_iter_per_epoch = no_samples//batch
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        if shuffle:
            random_index = torch.randperm(no_samples)
            x_train = x_train[random_index]
            y_train = y_train[random_index]
        for j in range(no_iter_per_epoch):
            if j != no_iter_per_epoch - 1:
                out = model(x_train[batch*j:batch*(j+1)])
                loss = loss_fn(out, y_train[batch*j:batch*(j+1)])
            else:
                out = model(x_train[batch*j:])
                loss = loss_fn(out, y_train[batch*j:])

            # Calculate gradient
            loss.backward()

            # Update weights
            optimizer.step()

            # Reset gradient
            optimizer.zero_grad()

            if verbose == 1 and j == no_iter_per_epoch - 1:
                out = model(x_train)
                out[out>0.5] = 1
                out[out<=0.5] = 0
                acc = (out == y_train).sum() / x_train.shape[0]
                print(f"Epoch {epoch+1}/{num_epochs}: loss = {loss.item(): 4f}, "
                      f"accuracy = {acc: 4f}")


def classifier_train_valid(model, x_train, y_train, x_valid, y_valid, lr=0.01,
                           num_epochs=20, batch=64, loss_fn=nn.BCELoss(),
                           patience=5, delta=0.01, shuffle=True, verbose=0):
    if not torch.is_tensor(x_train):
        x_train = torch.Tensor(x_train)
    if not torch.is_tensor(y_train):
        y_train = torch.Tensor(y_train)
    if not torch.is_tensor(x_valid):
        x_valid = torch.Tensor(x_valid)
    if not torch.is_tensor(y_valid):
        y_valid = torch.Tensor(y_valid)

    no_samples = x_train.shape[0]
    no_iter_per_epoch = no_samples//batch
    no_valid_samples = x_valid.shape[0]
    no_valid_loops = no_valid_samples // batch
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience, delta=delta)

    for epoch in range(num_epochs):
        train_loss_list = []
        if shuffle:
            random_index = torch.randperm(no_samples)
            x_train = x_train[random_index]
            y_train = y_train[random_index]
        for j in range(no_iter_per_epoch):
            if j != no_iter_per_epoch - 1:
                out = model(x_train[batch*j:batch*(j+1)])
                loss = loss_fn(out, y_train[batch*j:batch*(j+1)])
            else:
                out = model(x_train[batch*j:])
                loss = loss_fn(out, y_train[batch*j:])

            train_loss_list.append(loss)

            # Calculate gradient
            loss.backward()

            # Update weights
            optimizer.step()

            # Reset gradient
            optimizer.zero_grad()

            if j == no_iter_per_epoch - 1:
                # Validation
                valid_loss = 0.0
                for k in range(no_valid_loops):
                    if k != no_valid_loops - 1:
                        valid_out = model(x_valid[batch*k:batch*(k + 1)])
                        valid_loss = valid_loss + loss_fn(valid_out, y_valid[batch*k:batch*(k + 1)])
                    else:
                        valid_out = model(x_valid[batch*k:])
                        valid_loss = valid_loss + loss_fn(valid_out, y_valid[batch*k:])
                valid_loss = valid_loss / no_valid_loops
                early_stopping(val_loss=valid_loss, model=model)

                if verbose == 1:
                    train_out = model(x_train)
                    train_out[train_out>0.5] = 1
                    train_out[train_out<=0.5] = 0
                    train_acc = (train_out == y_train).sum() / x_train.shape[0]
                    train_loss = sum(train_loss_list)/len(train_loss_list)

                    valid_out = model(x_valid)
                    valid_out[valid_out>0.5] = 1
                    valid_out[valid_out<=0.5] = 0
                    valid_acc = (valid_out == y_valid).sum() / x_valid.shape[0]
                    print(f"Epoch {epoch+1}/{num_epochs}: train_loss = {train_loss: 4f}, "
                          f"train_acc = {train_acc: 4f}, valid_loss = {valid_loss: 4f}, "
                          f"valid_acc = {valid_acc: 4f}")
        if early_stopping.early_stop:
            model.load_state_dict(early_stopping.best_params)
            print("Early stopping.")
            break


def remove_frequency_component(x, freq_min, freq_max):
    """
    Function to remove frequencies out of the range.
    """

    try:
        x_fft = np.fft.rfft(x, axis=-1)

        x_fft[:, :, :freq_min] = 0
        x_fft[:, :, freq_max + 1:] = 0
        x_ifft = np.fft.irfft(x_fft)
        return x_ifft
    except:
        print("freq_min or freq_max is out of range.")