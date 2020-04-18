import os
from model.dkn import DKN
import requests
import zipfile
import io
import time
from utils import *
from config import *

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


dataset = {
    "train": read_dataset(os.path.join(DATASET_DIR, 'final_train_with_entity.txt')),
    "test": read_dataset(os.path.join(DATASET_DIR, 'final_test_with_entity.txt'))
}

# TODO
# pad dataset

print(
    f"Load dataset with train size {len(dataset['train'])} and test size {len(dataset['test'])}.")


dkn = DKN()

# TODO
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
start_time = time.time()
loss_full = []
for i in range(1, EPOCH_NUM + 1):
    y_pred = dkn(x)
    loss = criterion(y_pred, y)
    loss_full.append(loss.item())
    if i % 50 == 0:
        print(
            f"Epoch {i} ({time_since(start_time)}), batch loss {loss.item()}: , average loss: {np.mean(loss_full)}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
