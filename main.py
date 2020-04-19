import os
from model.dkn import DKN
from torch.utils.data import DataLoader
import torch
import time
import numpy as np
from config import Config
from dataset import DKNDataset

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
# TODO use cuda if possible

# TODO some layers in F should be use nn.* instead
# in order to be shown in model summary


def main():
    # TODO Fullfill train_dataset and test_dataset
    train_dataset = DKNDataset(
        Config)
    test_dataset = DKNDataset(
        Config)
    print(
        f"Load dataset with train size {len(train_dataset)} and test size {len(test_dataset)}."
    )

    train_dataloader = iter(
        DataLoader(train_dataset,
                   batch_size=Config.batch_size,
                   shuffle=True,
                   num_workers=Config.num_workers,
                   drop_last=True))

    # TODO
    embeddings = {
        # num_words_a_sentence, word_embedding_dim
        "word": 0,
        # num_entity_tokens, entity_embedding_dim
        "entity": 0,
        # num_entity_tokens, entity_embedding_dim
        "context": 0
    }

    dkn = DKN(Config, embeddings)
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
