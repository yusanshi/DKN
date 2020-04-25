from config import Config
import fire
import pandas as pd
import numpy as np
import sys
sys.path.append('./thirdparty/OpenKE/')
from openke.config import Trainer  # noqa E402
from openke.module.model import TransE  # noqa E402
from openke.module.loss import MarginLoss  # noqa E402
from openke.module.strategy import NegativeSampling  # noqa E402
from openke.data import TrainDataLoader  # noqa E402


def train():
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path='./data/kg/',
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0)

    # define the model
    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=Config.entity_embedding_dim,
        p_norm=1,
        norm_flag=True)

    # define the loss function
    model = NegativeSampling(
        model=transe,
        loss=MarginLoss(margin=5.0),
        batch_size=train_dataloader.get_batch_size()
    )

    # train the model
    trainer = Trainer(model=model, data_loader=train_dataloader,
                      train_times=1000, alpha=1.0, use_gpu=True)
    trainer.run()
    transe.save_checkpoint('./data/kg/transe.ckpt')


def generate():
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path='./data/kg/',
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0)

    # define the model
    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=Config.entity_embedding_dim,
        p_norm=1,
        norm_flag=True)

    transe.load_checkpoint('./data/kg/transe.ckpt')
    entity_embedding = transe.get_parameters()['ent_embeddings.weight']
    entity_embedding[0] = 0
    np.save('./data/kg/entity.npy', entity_embedding)
    context_embedding = np.empty_like(entity_embedding)
    context_embedding[0] = 0
    relation = pd.read_table(
        './data/sub_kg/triple2id.txt', header=None)[[0, 1]]
    entity = pd.read_table('./data/sub_kg/entity2name.txt',
                           header=None)[[0]].to_numpy().flatten()

    for e in entity:
        df = pd.concat([relation[relation[0] == e],
                        relation[relation[1] == e]])
        context = list(set(np.append(df.to_numpy().flatten(), e)))
        context_embedding[e] = np.mean(
            entity_embedding[context, :], axis=0)

    np.save('./data/kg/context.npy', context_embedding)


if __name__ == '__main__':
    fire.Fire()
