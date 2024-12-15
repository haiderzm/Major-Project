
## **Prerequisite**
"""

!pip install -q --no-cache-dir mne lightning torchmetrics

"""## **Important Libraries**"""

# Commented out IPython magic to ensure Python compatibility.
import mne
from mne.io import concatenate_raws

import os
import re
import io
import cv2
import random
import string
import warnings
import numpy as np
import matplotlib.pyplot as plt

from google.colab.patches import cv2_imshow

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable

import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from torchmetrics.classification import Accuracy , F1Score, Precision, Recall

# %matplotlib inline
# plt.rcParams['axes.facecolor'] = 'lightgray'

"""## **Dataset**"""

!gdown 19X1xKDECe7KDJqilGaKG3dvAA8oY52OJ

!unzip /content/data.zip

!rm -rf /content/data

!rm -rf /content/sample_data

# EEG Motor Movement/Imagery Dataset
# https://physionet.org/content/eegmmidb/1.0.0/
#
#
# This data set consists of over 1500 one- and two-minute EEG recordings,
# obtained from 109 volunteers.
#
# Subjects performed different motor/imagery tasks while 64-channel EEG were
# recorded using the BCI2000 system (http://www.bci2000.org).
#
# In summary, the experimental runs were:
# - (Baseline, eyes open)                                 ---> 1
# - (Baseline, eyes closed)                               ---> 2
# - (open and close the left or right fist)               ---> 3,7,11
# - (imagine opening and closing the left or right fist)  ---> 4,8,12
# - (open and close both fists or both feet)              ---> 5,9,13
# - (imagine opening and closing both fists or both feet) ---> 6,10,14
#
# Each annotation includes one of three codes of events (T0=1, T1=2, or T2=3):
# - T0 corresponds to rest
# - T1 corresponds to onset of motion (real or imagined) of
#     > the left fist (in runs 3, 4, 7, 8, 11, and 12)
#     > both fists (in runs 5, 6, 9, 10, 13, and 14)
# - T2 corresponds to onset of motion (real or imagined) of
#     > the right fist (in runs 3, 4, 7, 8, 11, and 12)
#     > both feet (in runs 5, 6, 9, 10, 13, and 14)
################################################################################


warnings.filterwarnings("ignore")


N_SUBJECT = 109
BASELINE_EYE_OPEN = [1]
BASELINE_EYE_CLOSED = [2]
OPEN_CLOSE_LEFT_RIGHT_FIST = [3, 7, 11]
IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST = [4, 8, 12]
OPEN_CLOSE_BOTH_FIST = [5, 9, 13]
IMAGINE_OPEN_CLOSE_BOTH_FIST = [6, 10, 14]


physionet_paths = [
    mne.datasets.eegbci.load_data(
        id,
        IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST,
        "/content/content/data",
    )
    for id in range(1, (N_SUBJECT - 30) + 1) # pick 1 to 79
]
physionet_paths = np.concatenate(physionet_paths)


# Read more:
# Tutorial #1: Loading EEG Data
# https://neuro.inf.unibe.ch/AlgorithmsNeuroscience/Tutorial_files/DataLoading.html



# size: 237 x 4680320 x 2 x 1 x 19680
# -----------------------------------
# 237 = 79 * 3 [num_subject * 3] -> 3 runs [4, 8, 12]
# 4680320 -> raw time (?)
parts = [
    mne.io.read_raw_edf(
        path,
        preload=True,
        stim_channel='auto',
        verbose='WARNING',
    )
    for path in physionet_paths
]


# size: (64, 4680320) -> channels x raw times
raw = concatenate_raws(parts)
sample_raw_data = raw.get_data()[0,:500]


# size: (7110, 3) -> 3 type of events (?)
# -----------------------
# 7110 = 237 * 30 = 79 (subjects) * 3 (runs) * 30 -> per runs 30 events(?)
events, _ = mne.events_from_annotations(raw)


# size: 64 -> EEG-channel
eeg_channel_inds = mne.pick_types(
    raw.info,
    meg=False,
    eeg=True,
    stim=False,
    eog=False,
    exclude='bads',
)


EEG_CHANNEL = int(len(eeg_channel_inds))


# epoched data: (3377, 64, 497) -> events x channel x time instances
# epoched events: (3377, 3) -> events x class
epoched = mne.Epochs(
    raw,
    events,
    dict(left=2, right=3),
    tmin=1,
    tmax=4.1,
    proj=False,
    picks=eeg_channel_inds,
    baseline=None,
    preload=True
)


# What does epoch mean in EEG?
# https://dsp.stackexchange.com/questions/41135/what-does-epoch-mean-in-eeg
#
# EEG epoching is a procedure in which specific time-windows are extracted from
# the continuous EEG signal. These time windows are called “epochs”, and usually
# are time-locked with respect an event e.g. a visual stimulus.
#
# If your EEG data are in a matrix [channel x time] where time is the complete
# continuous EEG signal, after the epoching procedure you should have a matrix
# [channel x time x epochs] where time is the time length of each epoch, and
# epochs is the number of segments you extracted from continuous EEG signal.
#
# Finally, if you want to extract epochs from your signal, you should know what
# are the segments of interest to be analyzed, for instance, a specific stimulus.


# Source:
# https://github.com/DavidSilveraGabriel/EEG-classification/blob/master/Using_mne_and_braindecode.ipynb

# Convert data from Volt to milliVolt
# size = (3377, 64, 497) -> [epochs or event x channel x time instance]
X = (epoched.get_data() * 1e3).astype(np.float32)

# The 0 represent the left and the 1 represent the right
# Source: https://mne.tools/stable/glossary.html#term-events
# Events correspond to specific time points in raw data, such as triggers,
# experimental condition events, etc. MNE-Python represents events with integers
# stored in NumPy arrays of shape (n_events, 3).
# The first column contains the event onset (in samples) with first_samp included.
# The last column contains the event code. <--- IMPORTANT TO NOTE
# The second column contains the signal value of the immediately preceding sample,
# and reflects the fact that event arrays sometimes originate from
# analog voltage channels (“trigger channels” or “stim channels”).
# In most cases, the second column is all zeros and can be ignored.
# size = (3377, )
y = (epoched.events[:, 2] - 2).astype(np.int64)


CLASSES = ["left", "right"]

"""## Definning Dataset class"""

class EEGDataset(data.Dataset):
    def __init__(self, x, y=None, inference=False):
        super().__init__()

        N_SAMPLE = x.shape[0]
        val_idx = int(0.9 * N_SAMPLE)
        train_idx = int(0.81 * N_SAMPLE)

        if not inference:
            self.train_ds = {
                'x': x[:train_idx],
                'y': y[:train_idx],
            }
            self.val_ds = {
                'x': x[train_idx:val_idx],
                'y': y[train_idx:val_idx],
            }
            self.test_ds = {
                'x': x[val_idx:],
                'y': y[val_idx:],
            }
        else:
            self.__split = "inference"
            self.inference_ds = {
                'x': [x],
            }

    def __len__(self):
        return len(self.dataset['x'])

    def __getitem__(self, idx):

        x = self.dataset['x'][idx]
        if self.__split != "inference":
            y = self.dataset['y'][idx]
            x = torch.tensor(x).float()
            y = torch.tensor(y).unsqueeze(-1).float()
            return x, y
        else:
            x = torch.tensor(x).float()
            return x

    def split(self, __split):
        self.__split = __split
        return self

    @classmethod
    def inference_dataset(cls, x):
        return cls(x, inference=True)

    @property
    def dataset(self):
        assert self.__split is not None, "Please specify the split of dataset!"

        if self.__split == "train":
            return self.train_ds
        elif self.__split == "val":
            return self.val_ds
        elif self.__split == "test":
            return self.test_ds
        elif self.__split == "inference":
            return self.inference_ds
        else:
            raise TypeError("Unknown type of split!")

eeg_dataset = EEGDataset(x=X, y=y)

plt.plot(sample_raw_data)
plt.title("Raw EEG, electrode 0, samples 0-500")
plt.ylabel("mV")
plt.xlabel("Sample")
plt.show()
plt.clf()

plt.plot(X[18:21, 0, :].T)
plt.title("Exemplar single-trial epoched data, for electrode 0")
plt.ylabel("V")
plt.xlabel("Epoched Sample")
plt.show()
plt.clf()

# RAM saving
del raw, events, epoched, physionet_paths, eeg_channel_inds, parts

device = 'cuda' if torch.cuda.is_available() else 'cpu'



"""#

## Pretext Task: Contrastive Model & training
"""

class PositionalEncoding(nn.Module):
    """Positional encoding.
    https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html
    """
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.p = torch.zeros((1, max_len, num_hiddens))
        x = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.p[:, :, 0::2] = torch.sin(x)
        self.p[:, :, 1::2] = torch.cos(x)

    def forward(self, x):
        x = x + self.p[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
        )

        self.layernorm0 = nn.LayerNorm(embed_dim)
        self.layernorm1 = nn.LayerNorm(embed_dim)

        self.dropout = dropout

    def forward(self, x):
        y, att = self.attention(x, x, x)
        y = F.dropout(y, self.dropout, training=self.training)
        x = self.layernorm0(x + y)
        y = self.mlp(x)
        y = F.dropout(y, self.dropout, training=self.training)
        x = self.layernorm1(x + y)
        return x

class EEGClassificationModel(nn.Module):
    def __init__(self, eeg_channel, dropout=0.1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                eeg_channel, eeg_channel, 11, 1, padding=5, bias=False
            ),
            nn.BatchNorm1d(eeg_channel),
            nn.ReLU(True),
            nn.Dropout1d(dropout),
            nn.Conv1d(
                eeg_channel, eeg_channel * 2, 11, 1, padding=5, bias=False
            ),
            nn.BatchNorm1d(eeg_channel * 2),
        )

        self.transformer = nn.Sequential(
            PositionalEncoding(eeg_channel * 2, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
        )

        self.mlp = nn.Sequential(
            nn.Linear(eeg_channel * 2, eeg_channel // 2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(eeg_channel // 2, 128),
        )

    def forward(self, x):
        print("shape before cnn: ", x.shape)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        print("shape before Transformer:",x.shape)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = x.mean(dim=-1)
        x = self.mlp(x)
        return x

model = ContrastiveEEGModel(eeg_channel=64).to(device)

!gdown 1UVvQFQjuzQlb1KNdKyn40HbbnY_7OS4D

model.load_state_dict(torch.load('/content/contrastive_best.pth'))

import torch
import torch.nn as nn
import torch.nn.functional as F
num_epochs = 100
from tqdm import tqdm
# Assuming you have the EEGClassificationModel defined in model.py

# Define data augmentation functions
def random_mask(x, mask_ratio=0.2):
    """Apply random masking to the input."""
    mask = torch.rand(x.shape, device=x.device) < mask_ratio
    x_masked = x.clone()
    x_masked[mask] = 0
    return x_masked
def gaussian_noise(x, std=0.1):
    noise = torch.randn_like(x) * std
    return x + noise
def random_shift(x, shift_ratio=0.1):
    """Apply random temporal shifting to the input."""
    shift = int(x.shape[-1] * shift_ratio * torch.rand(1)[0])
    x_shifted = torch.roll(x, shifts=shift, dims=-1)
    return x_shifted

class ContrastiveEEGModel(nn.Module):
    def __init__(self, eeg_channel, projection_dim=128, dropout=0.1):
        super().__init__()
        self.eeg_model = EEGClassificationModel(eeg_channel, dropout)
        self.projection_head = nn.Sequential(
            nn.Linear(eeg_channel * 2, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(True),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x):
        z1 = self.eeg_model(x)
        z2 = self.eeg_model(gaussian_noise(random_mask(x)))
        h1 = self.projection_head(z1)
        h2 = self.projection_head(z2)
        return h1, h2

def contrastive_loss(h1, h2, temperature=0.5):
    """Contrastive loss function."""
    batch_size = h1.shape[0]
    sim_matrix = F.cosine_similarity(h1.unsqueeze(1), h2.unsqueeze(0), dim=-1)
    sim_matrix = sim_matrix / temperature

    sim_ij = torch.diag(sim_matrix, 0)
    sim_ji = torch.diag(sim_matrix, -1)
    positives = torch.cat([sim_ij, sim_ji], dim=0)

    numerator = torch.exp(positives)
    denominator = numerator.sum(dim=0)

    loss = -torch.log(numerator / denominator).mean()
    return loss

# # Training loop
train_dataset = EEGDataset(X, y).split("train")
val_dataset = EEGDataset(X, y).split("val")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

# model = ContrastiveEEGModel(eeg_channel=64).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

# prev_loss = 1000
# for epoch in tqdm(range(100)):
#     model.train()
#     for x, _ in train_loader:
#         x = x.to(device)
#         h1, h2 = model(x)
#         loss = contrastive_loss(h1, h2)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     model.eval()
#     with torch.no_grad():
#         val_loss = 0
#         for x, _ in val_loader:
#             x = x.to(device)
#             h1, h2 = model(x)
#             val_loss += contrastive_loss(h1, h2).item()
#         val_loss /= len(val_loader)

#         if val_loss <prev_loss:
#           torch.save(model.state_dict(), 'contrastive_best.pth')
#           prev_loss = val_loss

#     print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

for b, (data,label) in enumerate(train_loader):
  print(data[0,0,:].shape)
  plt.plot(data[0,0,:])
  break

plt.figure(figsize=(40,8))
plt.plot(random_mask(data[0,0,:200]))

plt.figure(figsize=(40,8))
plt.plot(random_mask(data[0,0,:200]))

!pip install -q torchinfo
from torchinfo import summary

print(summary(model,input_size=[10,64,397]))

model

"""## **Model**

### **Utils**
"""

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.losses = []

    def update(self, val):
        self.losses.append(val)

    def show(self):
        out = torch.mean(
            torch.stack(
                self.losses[np.maximum(len(self.losses)-self.num, 0):]
            )
        )
        return out

"""### **Wrapper**"""

class ModelWrapper(L.LightningModule):
    def __init__(self, arch, dataset, batch_size, lr, max_epoch):
        super().__init__()

        self.arch = arch
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr
        self.max_epoch = max_epoch

        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.test_accuracy = Accuracy(task="binary")

        self.train_precision = Precision(task="binary")
        self.val_precision = Precision(task="binary")

        self.train_recall = Recall(task="binary")
        self.val_recall = Recall(task="binary")

        self.train_f1_score = F1Score(task="binary")
        self.val_f1_score = F1Score(task="binary")


        # F1Score, Precision, Recall

        self.automatic_optimization = False

        self.train_loss = []
        self.val_loss = []

        self.train_acc = []
        self.val_acc = []

        self.train_pres = []
        self.val_pres = []

        self.train_rec = []
        self.val_rec = []

        self.train_f1 = []
        self.val_f1 = []

        self.train_loss_recorder = AvgMeter()
        self.val_loss_recorder = AvgMeter()

        self.train_acc_recorder = AvgMeter()
        self.val_acc_recorder = AvgMeter()

        self.train_pres_recorder = AvgMeter()
        self.val_pres_recorder = AvgMeter()

        self.train_recall_recorder = AvgMeter()
        self.val_recall_recorder = AvgMeter()

        self.train_f1_recorder = AvgMeter()
        self.val_f1_recorder = AvgMeter()

    def forward(self, x):
        return self.arch(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat,_ = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.train_accuracy.update(y_hat, y)
        self.train_precision.update(y_hat, y)
        self.train_recall.update(y_hat, y)
        self.train_f1_score.update(y_hat, y)

        acc = self.train_accuracy.compute().data.cpu()
        p = self.train_precision.compute().data.cpu()
        r = self.train_recall.compute().data.cpu()
        f1 = self.train_f1_score.compute().data.cpu()

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        self.train_loss_recorder.update(loss.data)
        self.train_acc_recorder.update(acc)
        self.train_pres_recorder.update(p)
        self.train_recall_recorder.update(r)
        self.train_f1_recorder.update(f1)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_pr",p,prog_bar=True)
        self.log("train_rec",r,prog_bar=True)
        self.log("train_f1",f1,prog_bar=True)

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        sch.step()

        self.train_loss.append(self.train_loss_recorder.show().data.cpu().numpy())
        self.train_loss_recorder = AvgMeter()

        self.train_acc.append(self.train_acc_recorder.show().data.cpu().numpy())
        self.train_acc_recorder = AvgMeter()

        self.train_pres.append(self.train_pres_recorder.show().data.cpu().numpy())
        self.train_pres_recorder = AvgMeter()

        self.train_rec.append(self.train_recall_recorder.show().data.cpu().numpy())
        self.train_recall_recorder = AvgMeter()

        self.train_f1.append(self.train_f1_recorder.show().data.cpu().numpy())
        self.train_f1_recorder = AvgMeter()



    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat,_ = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.val_accuracy.update(y_hat, y)
        self.val_precision.update(y_hat, y)
        self.val_recall.update(y_hat, y)
        self.val_f1_score.update(y_hat, y)

        acc = self.val_accuracy.compute().data.cpu()
        p = self.val_precision.compute().data.cpu()
        r = self.val_recall.compute().data.cpu()
        f1 = self.val_f1_score.compute().data.cpu()

        self.val_loss_recorder.update(loss.data)
        self.val_acc_recorder.update(acc)
        self.val_pres_recorder.update(p)
        self.val_recall_recorder.update(r)
        self.val_f1_recorder.update(f1)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_pr",p,prog_bar=True)
        self.log("val_rec",r,prog_bar=True)
        self.log("val_f1",f1,prog_bar=True)

    def on_validation_epoch_end(self):
        self.val_loss.append(self.val_loss_recorder.show().data.cpu().numpy())
        self.val_loss_recorder = AvgMeter()

        self.val_acc.append(self.val_acc_recorder.show().data.cpu().numpy())
        self.val_acc_recorder = AvgMeter()

        self.val_pres.append(self.val_pres_recorder.show().data.cpu().numpy())
        self.val_pres_recorder = AvgMeter()

        self.val_rec.append(self.val_recall_recorder.show().data.cpu().numpy())
        self.val_recall_recorder = AvgMeter()

        self.val_f1.append(self.val_f1_recorder.show().data.cpu().numpy())
        self.val_f1_recorder = AvgMeter()

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.test_accuracy.update(y_hat, y)

        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_acc",
            self.test_accuracy.compute(),
            prog_bar=True,
            logger=True,
        )

    def on_train_end(self):

        # Loss
        np.save('train_loss.npy', self.train_loss)
        np.save('val_loss.npy', self.val_loss)

        np.save("train_p.npy",self.train_pres)
        np.save("train_r.npy",self.train_rec)
        np.save("train_f1.npy",self.train_f1)

        np.save("val_p.npy",self.val_pres)
        np.save("val_r.npy",self.val_rec)
        np.save("val_f1.npy",self.val_f1)

        loss_img_file = "/content/loss_plot.png"
        plt.plot(self.train_loss, color = 'r', label='train')
        plt.plot(self.val_loss, color = 'b', label='validation')
        plt.title("Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(loss_img_file)
        plt.clf()
        img = cv2.imread(loss_img_file)
        cv2_imshow(img)

        # Accuracy
        np.save('train_acc.npy', self.train_acc)
        np.save('val_acc.npy', self.val_acc)
        acc_img_file = "/content/acc_plot.png"
        plt.plot(self.train_acc, color = 'r', label='train')
        plt.plot(self.val_acc, color = 'b', label='validation')
        plt.title("Accuracy Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid()
        plt.savefig(acc_img_file)
        plt.clf()
        img = cv2.imread(acc_img_file)
        cv2_imshow(img)

    def train_dataloader(self):
        return data.DataLoader(
            dataset=self.dataset.split("train"),
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            dataset=self.dataset.split("val"),
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return data.DataLoader(
            dataset=self.dataset.split("test"),
            batch_size=1,
            shuffle=False,
        )

    def configure_optimizers(self):

        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
        )
        lr_scheduler = {
            "scheduler": optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    int(self.max_epoch * 0.25),
                    int(self.max_epoch * 0.5),
                    int(self.max_epoch * 0.75),
                ],
                gamma=0.1
            ),
            "name": "lr_scheduler",
        }
        return [optimizer], [lr_scheduler]

"""# DownStream Task Training"""



"""### **EEG Classification Model**"""

class PositionalEncoding(nn.Module):
    """Positional encoding.
    https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html
    """
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.p = torch.zeros((1, max_len, num_hiddens))
        x = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.p[:, :, 0::2] = torch.sin(x)
        self.p[:, :, 1::2] = torch.cos(x)

    def forward(self, x):
        x = x + self.p[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
        )

        self.layernorm0 = nn.LayerNorm(embed_dim)
        self.layernorm1 = nn.LayerNorm(embed_dim)

        self.dropout = dropout

    def forward(self, x):
        y, att = self.attention(x, x, x)
        y = F.dropout(y, self.dropout, training=self.training)
        x = self.layernorm0(x + y)
        y = self.mlp(x)
        y = F.dropout(y, self.dropout, training=self.training)
        x = self.layernorm1(x + y)
        return x

class EEGClassificationModel(nn.Module):
    def __init__(self, eeg_channel, dropout=0.1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                eeg_channel, eeg_channel, 11, 1, padding=5, bias=False
            ),
            nn.BatchNorm1d(eeg_channel),
            nn.ReLU(True),
            nn.Dropout1d(dropout),
            nn.Conv1d(
                eeg_channel, eeg_channel * 2, 11, 1, padding=5, bias=False
            ),
            nn.BatchNorm1d(eeg_channel * 2),
        )

        self.transformer = nn.Sequential(
            PositionalEncoding(eeg_channel * 2, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
        )

        self.mlp = nn.Sequential(
            nn.Linear(eeg_channel * 2, eeg_channel // 2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(eeg_channel // 2, 1),
        )

    def forward(self, x):
        print("shape before cnn: ", x.shape)





        x = self.conv(x)
        x = x.permute(0, 2, 1)
        print("shape before Transformer:",x.shape)
        x = self.transformer(x)

        x = x.permute(0, 2, 1)
        x = x.mean(dim=-1)
        x = self.mlp(x)
        return x

model.eeg_model.mlp = nn.Sequential(
            nn.Linear(EEG_CHANNEL * 2, EEG_CHANNEL // 2),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(EEG_CHANNEL // 2, 1),
        )

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

model.projection_head = Identity()

MODEL_NAME = "EEGClassificationModel"
model = EEGClassificationModel(eeg_channel=EEG_CHANNEL, dropout=0.14)

model

"""## **Training**"""

MAX_EPOCH = 500
BATCH_SIZE = 10
LR = 5e-4
CHECKPOINT_DIR = os.getcwd()
SEED = int(np.random.randint(2147483647))

print(f"Random seed: {SEED}")

model = ModelWrapper(model, eeg_dataset, BATCH_SIZE, LR, MAX_EPOCH)

!rm -rf logs/

# Commented out IPython magic to ensure Python compatibility.
# %reload_ext tensorboard
# %tensorboard --logdir=logs/lightning_logs/

tensorboardlogger = TensorBoardLogger(save_dir="logs/")
csvlogger = CSVLogger(save_dir="logs/")
lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint = ModelCheckpoint(
    monitor='val_acc',
    dirpath=CHECKPOINT_DIR,
    mode='max',
)
early_stopping = EarlyStopping(
    monitor="val_acc", min_delta=0.00, patience=3, verbose=False, mode="max"
)


seed_everything(SEED, workers=True)


trainer = Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=MAX_EPOCH,
    logger=[tensorboardlogger, csvlogger],
    callbacks=[lr_monitor, checkpoint],
    log_every_n_steps=5,
)
trainer.fit(model)

!mkdir epoch_200/

!mkdir epoch_500/

!cp *.npy epoch_200/

!cp *.npy epoch_500/

!cp *.png epoch_500/

!zip -r epoch_500.zip epoch_500/

"""## **Testing**"""

trainer.test(ckpt_path="best")

os.rename(
    checkpoint.best_model_path,
    os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.ckpt")
)

inp = torch.rand([10,64,397])
out = model(inp)
out

model

!cp /content/data.zip /content/drive/MyDrive/eeg/

"""## **Inference**"""

for _ in range(5):
    N_SAMPLE = X.shape[0]
    sample_idx = random.randint(0, N_SAMPLE - 1)
    sample = X[sample_idx]

    trainer = Trainer()
    prediction = trainer.predict(
        model=model,
        dataloaders=data.DataLoader(
            dataset=EEGDataset.inference_dataset(X[sample_idx]),
            batch_size=1,
            shuffle=False,
        ),
        ckpt_path=os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.ckpt"),
    )[0]

    PREDICTED = CLASSES[int(torch.sigmoid(prediction) > 0.5)]
    ACTUAL = CLASSES[y[sample_idx]]
    print("\n\n\n")
    print(f"Imagining {PREDICTED} hand movement!")
    print(f"Ground-truth: {ACTUAL}!")

    plt.plot(sample.T)
    plt.title(
        f"Exemplar of epoched data, for electrode 0-63\nActual Label : {ACTUAL}\nPredicted Label : {PREDICTED}"
    )
    plt.ylabel("V")
    plt.xlabel("Epoched Sample")
    plt.show()
    plt.clf()

    print("\n\n\n")

