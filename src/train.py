from model.dkn import DKN
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import time
import numpy as np
from config import Config
from dataset import DKNDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from evaluate import ndcg_score, mrr_score
import os


def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    all_checkpoints = {
        int(x.split('.')[-2].split('-')[-1]): x
        for x in os.listdir(directory)
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])


def train():
    print(f"Context: {Config.use_context}, Attention: {Config.use_attention}")
    writer = SummaryWriter(
        comment=f"Context-{Config.use_context}_Attention-{Config.use_attention}"
    )

    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')

    dataset = DKNDataset('data/train/behaviors_cleaned.tsv',
                         'data/train/news_with_entity.tsv')
    train_size = int(Config.train_validation_split[0] /
                     sum(Config.train_validation_split) * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset,
                                              (train_size, validation_size))
    print(
        f"Load training dataset with train size {len(train_dataset)} and validation size {len(val_dataset)}."
    )

    train_dataloader = iter(
        DataLoader(train_dataset,
                   batch_size=Config.batch_size,
                   shuffle=True,
                   num_workers=Config.num_workers,
                   drop_last=True))

    # Load trained embedding file
    # num_entity_tokens, entity_embedding_dim
    entity_embedding = np.load('data/train/entity_embedding.npy')
    context_embedding = np.load('data/train/entity_embedding.npy')  # TODO

    dkn = DKN(Config, entity_embedding, context_embedding).to(device)
    print(dkn)

    # TODO magic number 23.7
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([23.7]).float().to(device))
    optimizer = torch.optim.Adam(dkn.parameters(), lr=Config.learning_rate)
    start_time = time.time()
    loss_full = []
    exhaustion_count = 0
    epoch = 0

    if Config.load_checkpoint:
        checkpoint_path = latest_checkpoint('./checkpoint')
        if checkpoint_path is not None:
            print(f"Load saved parameters in {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            dkn.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            dkn.train()

    with tqdm(total=Config.num_batches, desc="Training") as pbar:
        for i in range(1, Config.num_batches + 1):
            try:
                minibatch = next(train_dataloader)
            except StopIteration:
                exhaustion_count += 1
                tqdm.write(
                    f"Training data exhausted for {exhaustion_count} times after {i} batches, reuse the dataset."
                )
                train_dataloader = iter(
                    DataLoader(train_dataset,
                               batch_size=Config.batch_size,
                               shuffle=True,
                               num_workers=Config.num_workers,
                               drop_last=True))
                minibatch = next(train_dataloader)

            epoch += 1

            y_pred = dkn(minibatch["candidate_news"],
                         minibatch["clicked_news"])
            y = minibatch["clicked"].float().to(device)
            loss = criterion(y_pred, y)
            loss_full.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Train/Loss', loss.item(), i)

            if i % Config.num_batches_batch_loss == 0:
                tqdm.write(
                    f"Time {time_since(start_time)}, batches {i}, current loss {loss.item():.6f}, average loss: {np.mean(loss_full):.6f}"
                )

            if i % Config.num_batches_val_loss_and_acc == 0:
                val_loss, val_auc, val_mrr, val_ndcg5, val_ncg10 = validate(
                    dkn, val_dataset)
                writer.add_scalar('Validation/Loss', val_loss, i)
                writer.add_scalar('Validation/AUC', val_auc, i)
                writer.add_scalar('Validation/MRR', val_mrr, i)
                writer.add_scalar('Validation/nDCG@5', val_ndcg5, i)
                writer.add_scalar('Validation/nDCG@10', val_ncg10, i)
                tqdm.write(
                    f"Time {time_since(start_time)}, batches {i}, validation loss: {val_loss:.6f}, validation AUC: {val_auc:.6f}, validation MRR: {val_mrr:.6f}, validation nDCG@5: {val_ndcg5:.6f}, validation nDCG@10: {val_ncg10:.6f}, "
                )

            if i % Config.num_batches_save_checkpoint == 0:
                torch.save(
                    {
                        'model_state_dict': dkn.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch
                    }, f"./checkpoint/ckpt-{epoch}.pth")

            pbar.update(1)

    torch.save(
        {
            'model_state_dict': dkn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }, f"./checkpoint/ckpt-{epoch}.pth")

    val_loss, val_auc, val_mrr, val_ndcg5, val_ncg10 = validate(
        dkn, val_dataset)
    writer.add_scalar('Validation/Loss', val_loss, Config.num_batches)
    writer.add_scalar('Validation/AUC', val_auc, Config.num_batches)
    writer.add_scalar('Validation/MRR', val_mrr, Config.num_batches)
    writer.add_scalar('Validation/nDCG@5', val_ndcg5, Config.num_batches)
    writer.add_scalar('Validation/nDCG@10', val_ncg10, Config.num_batches)
    print(
        f"Final result on validation dataset, validation loss: {val_loss:.6f}, validation AUC: {val_auc:.6f}, validation MRR: {val_mrr:.6f}, validation nDCG@5: {val_ndcg5:.6f}, validation nDCG@10: {val_ncg10:.6f}, "
    )


@torch.no_grad()
def validate(model, dataset):
    """
    Check loss and accuracy of trained model on given validation dataset.
    """
    dataloader = DataLoader(dataset,
                            batch_size=Config.batch_size,
                            shuffle=True,
                            num_workers=Config.num_workers,
                            drop_last=True)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([23.7]).float().to(device))
    loss_full = []
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []
    with tqdm(total=len(dataloader),
              desc="Checking loss and accuracy") as pbar:
        for minibatch in dataloader:
            y_pred = model(minibatch["candidate_news"],
                           minibatch["clicked_news"])
            y = minibatch["clicked"].float().to(device)
            loss = criterion(y_pred, y)
            loss_full.append(loss.item())
            y_pred_list = y_pred.tolist()
            y_list = y.tolist()

            auc = roc_auc_score(y_list, y_pred_list)
            mrr = mrr_score(y_list, y_pred_list)
            ndcg5 = ndcg_score(y_list, y_pred_list, 5)
            ndcg10 = ndcg_score(y_list, y_pred_list, 10)

            aucs.append(auc)
            mrrs.append(mrr)
            ndcg5s.append(ndcg5)
            ndcg10s.append(ndcg10)

            pbar.update(1)

    return np.mean(loss_full), np.mean(aucs), np.mean(mrrs), np.mean(
        ndcg5s), np.mean(ndcg10s)


def time_since(since):
    """
    Format elapsed time string.
    """
    now = time.time()
    elapsed_time = now - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


if __name__ == '__main__':
    # setting device on GPU if available, else CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    train()
