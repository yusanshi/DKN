import os
from model.dkn import DKN
from torch.utils.data import DataLoader
import torch
import requests
import zipfile
import io
import time
import numpy as np
from config import Config
from dataset import DKNDataset


def main():
    # TODO:cuda device = ? else

    # Download dataset if not exists
    DATASET_DIR = './dataset'
    if not os.path.isdir(DATASET_DIR):
        os.mkdir(DATASET_DIR)
        dkn_dataset_url = 'https://recodatasets.blob.core.windows.net/deeprec/dknresources.zip'
        r = requests.get(dkn_dataset_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(DATASET_DIR)
        print(f'Dataset downloaded into {DATASET_DIR}')
    else:
        print(f'Dataset exists in {DATASET_DIR}. Skip downloading.')

    # TODO change to final dataset
    train_dataset = DKNDataset(
        Config, os.path.join(DATASET_DIR, 'debug_train_with_entity.txt'))
    test_dataset = DKNDataset(
        Config, os.path.join(DATASET_DIR, 'debug_test_with_entity.txt'))
    print(
        f"Load dataset with train size {len(train_dataset)} and test size {len(test_dataset)}."
    )

    train_dataloader = iter(
        DataLoader(train_dataset,
                   batch_size=Config.batch_size,
                   shuffle=True,
                   num_workers=Config.num_workers,
                   drop_last=True))

    dkn = DKN(Config)
    # TODO
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(dkn.parameters(), lr=Config.learning_rate)
    start_time = time.time()
    loss_full = []
    exhaustion_count = 0

    for i in range(1, Config.num_batches + 1):
        try:
            minibatch = next(train_dataloader)
            y_pred = dkn(minibatch["candidatae_news"],
                         minibatch["clicked_news"])
            y = minibatch["clicked"].float()
            loss = criterion(y_pred, y)
            loss_full.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % Config.num_batches_batch_loss == 0:
                print(
                    f"Time {time_since(start_time)}, batches {i}, current loss {loss.item()}: , average loss: {np.mean(loss_full)}"
                )

            if i % Config.num_batches_val_loss == 0:
                print(
                    f"\nTime {time_since(start_time)}, batches {i}, validation loss: {check_loss(dkn, test_dataset)}\n"
                )

        except StopIteration:
            exhaustion_count += 1
            print(
                f"Training data exhausted for {exhaustion_count} times after {i} batches. Reuse the dataset."
            )
            train_dataloader = iter(
                DataLoader(train_dataset,
                           batch_size=Config.batch_size,
                           shuffle=True,
                           num_workers=Config.num_workers,
                           drop_last=True))


def check_loss(model, dataset):
    """
    Check average loss of trained model on given dataset.
    """
    dataloader = DataLoader(dataset,
                            batch_size=Config.batch_size,
                            shuffle=True,
                            num_workers=Config.num_workers,
                            drop_last=True)
    # TODO
    criterion = torch.nn.MSELoss()
    loss_full = []
    # TODO: not grad
    for minibatch in dataloader:
        y_pred = model(minibatch["candidatae_news"], minibatch["clicked_news"])
        y = minibatch["clicked"].float()
        loss = criterion(y_pred, y)
        loss_full.append(loss.item())
    return np.mean(loss_full)


def time_since(since):
    """
    Format elapsed time string.
    """
    now = time.time()
    elapsed_time = now - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


if __name__ == '__main__':
    main()
