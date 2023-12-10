import numpy as np
import pandas as pd
from PIL import Image
import torch
import tqdm
import matplotlib.pyplot as plt

# define function to standardize numerical features of dataframe
def standardize_numeric(series: pd.Series, use_log: bool = False) -> pd.Series:
    if use_log:
        series = np.log(series)
    return (series - np.mean(series))/np.std(series)

# define function to convert PIL images into correct format for exporting and model training
# Images are resized down to 64 x 64 pixels so that image processing does not exceed memory
# This further reduces time for image processing
def process_images(image_filenames: np.ndarray) -> torch.Tensor:
    
    images = []
    
    for image_path in image_filenames:
        im = Image.open('data/images/' + image_path)
        im = im.resize((64, 64))
        images.append(np.moveaxis(np.asarray(im), 2, 0))
    
    return torch.Tensor(np.asarray(images))

# define training loop function for model 2 and model 3
def training_loop(train_dataloader, val_dataloader, model, optimizer: torch.optim, epochs: int, loss_fn, train_losses, val_losses, is_model_2: bool=True):
    for _ in tqdm.tqdm(range(epochs)):
        losses = []
        for _ in range(train_dataloader.num_batches_per_epoch):
            # training data forward pass
            optimizer.zero_grad()
            train_batch = train_dataloader.fetch_batch()
            if is_model_2:
                yhat = model(train_batch['x_batch'])
            else:
                yhat = model(train_batch['x_batch_image'], train_batch['x_batch'])
            train_loss = torch.mean(loss_fn(yhat, train_batch['y_batch']), dim=0)

            # training data backward pass
            train_loss.backward()
            optimizer.step()
            losses.append(train_loss.detach().numpy())

        # personally, I like to visualize the loss per every iteration, rather than every epoch. I find it more useful to diagnose issues
        train_losses.extend(losses)

        losses = []
        for _ in range(val_dataloader.num_batches_per_epoch):
            # validation data forward pass only
            val_batch = val_dataloader.fetch_batch()
            if is_model_2:
                yhat = model(val_batch['x_batch'])
            else:
                yhat = model(val_batch['x_batch_image'], val_batch['x_batch'])
            val_loss = torch.mean(loss_fn(yhat, val_batch['y_batch']), dim=0)
            losses.append(val_loss.detach().numpy())
        # epoch-level logging for validation though usually makes the most sense
        val_losses.append(np.mean(losses))
    return train_losses, val_losses

# define function to plot training and validation losses
def plot_losses(train_losses, val_losses, epochs):
    plt.figure(0, figsize = (12,6))
    plt.title('Loss per iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(train_losses)
    plt.plot(np.linspace(0,len(train_losses),epochs), val_losses)
    plt.legend(["Training loss","Validation loss"])
    print('Epochs used: ', epochs)
    print('Final validation loss:',val_losses[-1], '\nNote that this is MSE loss.')
    print('The RMSE loss is ', np.sqrt(val_losses[-1]))

# define function to return model predictions that can be used to print loss
def run_inference(test_dataloader, model, is_model_2: bool = True) -> np.ndarray:
    y_preds = []
    for _ in range(test_dataloader.num_batches_per_epoch):
        test_batch = test_dataloader.fetch_batch()
        if is_model_2:
            yhat = model(test_batch['x_batch']).detach().numpy()
        else:
            yhat = model(test_batch['x_batch_image'], test_batch['x_batch']).detach().numpy()
        y_preds.append(yhat)
    y_preds = np.vstack(y_preds)
    return y_preds