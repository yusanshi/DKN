from model.dkn import DKN
import torch
from config import Config
import numpy as np
from tqdm import tqdm
from train import latest_checkpoint
from dataset import DKNDataset
from torch.utils.data import DataLoader
import json
import copy


def inference():
    dataset = DKNDataset('data/test/behaviors_cleaned.tsv',
                         'data/test/news_with_entity.tsv')
    print(f"Load inference dataset with size {len(dataset)}.")
    dataloader = DataLoader(dataset,
                            batch_size=Config.batch_size,
                            shuffle=False,
                            num_workers=Config.num_workers,
                            drop_last=False)

    # Load trained embedding file
    # num_entity_tokens, entity_embedding_dim
    entity_embedding = np.load('data/train/entity_embedding.npy')
    context_embedding = np.load('data/train/entity_embedding.npy')  # TODO

    dkn = DKN(Config, entity_embedding, context_embedding).to(device)
    checkpoint_path = latest_checkpoint('./checkpoint')
    print(f"Load saved parameters in {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    dkn.load_state_dict(checkpoint['model_state_dict'])
    dkn.eval()

    y_pred = []
    y = []
    with tqdm(total=len(dataloader), desc="Inferering") as pbar:
        for minibatch in dataloader:
            y_pred.extend(
                dkn(minibatch["candidate_news"],
                    minibatch["clicked_news"]).tolist())
            y.extend(minibatch["clicked"].float().tolist())
            pbar.update(1)

    y_pred = iter(y_pred)
    y = iter(y)

    # For locating and order validating
    truth_file = open('./data/test/truth.json', 'r')
    # For writing inference results
    submission_answer_file = open('./data/test/answer.json', 'w')
    try:
        for line in truth_file.readlines():
            user_truth = json.loads(line)
            user_inference = copy.deepcopy(user_truth)
            for k in user_truth['impression'].keys():
                assert next(y) == user_truth['impression'][k]
                user_inference['impression'][k] = next(y_pred)
            submission_answer_file.write(json.dumps(user_inference) + '\n')
    except StopIteration:
        print(
            'Warning: Behaviors not fully inferenced. You can still run evaluate.py, but the evaluation result would be inaccurate.'
        )

    submission_answer_file.close()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    inference()
