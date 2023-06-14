import json
import pathlib
import queue
from itertools import chain, islice

import clip
import pytorch_lightning as pl
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
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
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def main(file: pathlib.Path):

    suffix = file.suffix.lower()
    if suffix.endswith(".json"):
        meta = json.loads(file.read_text(encoding="utf-8"))
    else:
        raise Exception("Not metadata file?")

    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

    s = torch.load(
        "sac+logos+ava1-l14-linearMSE.pth"
    )  # load the model you trained previously or the model available in this repo

    model.load_state_dict(s)

    model.to("cuda")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model2, preprocess = clip.load("ViT-L/14", device=device)  # RN50x64
    qp = queue.Queue(maxsize=500)

    def qp_flush():
        with tqdm.tqdm(desc="Flush", total=qp.qsize()) as pbar:
            while not qp.empty():
                im_emb_arr, key = qp.get()

                print(len(im_emb_arr))

                prediction = model(
                    torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor)
                )
                value = round(prediction.item(), 4)
                if value >= 6.675:
                    value = "exceptional"
                elif value >= 6:
                    value = "best aesthetic"
                elif value >= 5.5:
                    value = "best aesthetic"  # Tempted to use "good aesthetic but i think it will mess up gen"
                elif value >= 5:
                    value = "normal aesthetic"
                elif value >= 4.5:
                    value = "bad aesthetic"  # Tempted to use "poor aesthetic but i think it will mess up gen"
                else:
                    value = "bad aesthetic"
                me_key = meta[key]
                if isinstance(me_key, list):
                    clone = {"tags": me_key, "clip_aesthetic": value}
                elif isinstance(me_key, dict):
                    clone = {**me_key, "clip_aesthetic": value}
                else:
                    raise Exception(f"Unknown type: {type(me_key)} for {key}: {me_key}")
                meta[key] = clone
                pbar.update(1)

    for key in tqdm.tqdm(meta.keys(), desc="Files done"):
        if qp.full():
            qp_flush()
        t_key = file.parent / key
        if meta.get(key,{}).get("clip_aesthetic"):
            continue
        with Image.open(t_key) as pil_image:
            if pil_image.mode == "RGBA":
                new_image = Image.new("RGBA", pil_image.size, "WHITE")
                new_image.paste(pil_image, mask=pil_image)
                new_image = new_image.convert("RGB")
            elif pil_image.info.get("transparency"):
                pil_image = pil_image.convert("RGBA")
                new_image = Image.new("RGBA", pil_image.size, "WHITE")
                new_image.paste(pil_image, mask=pil_image)
                new_image = new_image.convert("RGB")
            else:
                new_image = pil_image
            image = preprocess(new_image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model2.encode_image(image)
        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        qp.put((im_emb_arr, key))
    qp_flush()
    file.write_text(json.dumps(meta, indent=4, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    import sys

    main(pathlib.Path(sys.argv[1]))
