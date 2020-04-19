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
from kg import train_kg, load_kg

# TODO some layers in F should be use nn.* instead
# in order to be shown in model summary


def main():
    # TODO:cuda device = ? else

    dataset_dir = './dataset'
    # TODO change to final dataset
    train_file = 'debug_train_with_entity.txt'
    test_file = 'debug_test_with_entity.txt'
    # Download dataset if not exists
    if not (os.path.isfile(os.path.join(dataset_dir, train_file)) and os.path.isfile(os.path.join(dataset_dir, test_file))):
        print(f'Dataset not found in {dataset_dir}, start downloading.')
        dkn_dataset_url = 'https://yun.yusanshi.com/dkn_dataset.zip'
        r = requests.get(dkn_dataset_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(dataset_dir)
        print(f'Dataset downloaded and extracted into {dataset_dir}')
    else:
        print(f'Dataset exists in {dataset_dir}, skip downloading.')

    train_dataset = DKNDataset(
        Config, os.path.join(dataset_dir, train_file))
    test_dataset = DKNDataset(
        Config, os.path.join(dataset_dir, test_file))
    print(
        f"Load dataset with train size {len(train_dataset)} and test size {len(test_dataset)}."
    )

    kg_dir = './kg'
    kg_file = 'kg.json'  # TODO
    if not (os.path.isfile(os.path.join(kg_dir, kg_file))):
        print(f'Knowledge graph not found in {kg_dir}, start training.')
        train_kg(os.path.join(kg_dir, kg_file), Config.entity_embedding_dim)
        print(f'Knowledge graph trained and put into {kg_dir} for future use.')
    else:
        print(f'Knowledge graph exists in {kg_dir}, skip training.')

    entity_embedding, _ = load_kg(os.path.join(kg_dir, kg_file))

    train_dataloader = iter(
        DataLoader(train_dataset,
                   batch_size=Config.batch_size,
                   shuffle=True,
                   num_workers=Config.num_workers,
                   drop_last=True))

    dkn = DKN(Config, entity_embedding)
    print(dkn)
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
                    f"Time {time_since(start_time)}, batches {i}, current loss {loss.item()}, average loss: {np.mean(loss_full)}"
                )

            if i % Config.num_batches_val_loss == 0:
                print(
                    f"Time {time_since(start_time)}, batches {i}, validation loss: {check_loss(dkn, test_dataset)}"
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
