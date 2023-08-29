import clip
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # pylint: disable=import-outside-toplevel
from PIL import Image


# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol="emb", ycol="avg_rating"):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def normalized(a, axis=-1, order=2):
    l2 = torch.atleast_1d(torch.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / torch.unsqueeze(l2, axis)


class Tagger:
    def __init__(self) -> None:
        self.model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
        s = torch.load("sac+logos+ava1-l14-linearMSE.pth")
        self.model.load_state_dict(s)
        self.model.to("cuda")
        self.model.eval()
        self.model2, self.preprocess = clip.load("ViT-L/14", device="cuda")  # RN50x64

    def predict(self, image: Image.Image, scale: float = 10.0):
        image = self.preprocess(image).unsqueeze(0).to("cuda")
        with torch.no_grad():
            image_features = self.model2.encode_image(image)
        prediction = self.model(
            normalized(image_features.to("cuda").type(torch.cuda.FloatTensor))
        )
        value = round(prediction.item(), 4) * scale
        # print(value, prediction)
        return value