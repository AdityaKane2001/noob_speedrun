import torch 
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from model import ViT
from data import get_data_cnn
from utils import count_parameters

TR_BS = 2048
TE_BS = 1024
LR = 1e-8
EPOCHS = 10
NUM_LAYERS = 3 # 8 --> 10.9M, 4 --> 5.6M, 3 --> 4.3M
# effnet-b0 --> 4M

train_dl, test_dl = get_data_cnn(tr_bs=TR_BS, te_bs=TE_BS)
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# device = "cpu"
# model = ViT(num_layers=NUM_LAYERS).to(device)
# count_parameters(model)
print("device:", device)

from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=10) # 4M
count_parameters(model)
model.to(device)
print("__"*40)
# exit(0)

optim = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    running_loss = 0.
    correct = 0 
    total = 0
    labels = []
    preds = []
    for x, y in tqdm(train_dl, desc=f"Epoch: {epoch}"):
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optim.step()
        # Metrics
        labels.extend(y.detach().cpu().numpy())
        preds.extend(torch.argmax(out, dim=-1).detach().cpu().numpy())
        running_loss += loss.item()
        correct += (torch.argmax(out, dim=-1) == y).sum().item()
        total += y.shape[0]
    print(f"Epoch: {epoch} | Acc: {correct/total:.2f}, Loss: {running_loss/total:.2f}")
    print(confusion_matrix(labels, preds))


