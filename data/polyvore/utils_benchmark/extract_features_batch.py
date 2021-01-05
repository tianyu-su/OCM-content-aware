"""
Extract the features for each ssense image, using a resnet50 with pytorch
"""

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from multiprocessing import Pool
import os
import argparse
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage.color import gray2rgb, rgba2rgb
import skimage.io
from PIL import Image
import time
import pickle as pkl
import json

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-datadir', default='source', type=str)
parser.add_argument('-savedir', default='dest', type=str)
parser.add_argument('-bs', default='128', type=int)
parser.add_argument('-j', default='8', type=int)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load net to extract features
model = models.resnet50(pretrained=True)
# skip last layer (the classifier)
model = nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
model.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), normalize
])

json_file = os.path.join(args.datadir, 'polyvore_item_metadata.json')
with open(json_file) as f:
    meta_data = json.load(f)

save_to = args.savedir
if not os.path.exists(save_to):
    os.makedirs(save_to)
save_dict = os.path.join(save_to, 'benchmark_imgs_featdict.pkl')

# initialize empty feature matrix that will be filled
features = {}
print('iterating through ids')
n_items = len(meta_data.keys())

all_items = list(meta_data.keys())
batch_size = args.bs


def check_job(image_path):
    c_p = os.path.join(args.datadir, "images", f"{image_path}.jpg")
    if not os.path.exists(c_p):
        return image_path
    else:
        return False


class ExtDataset(torch.utils.data.Dataset):
    def __init__(self, pre_path, all_items):
        self.pre_path = pre_path
        self.items = self.check(all_items)

    @staticmethod
    def check(all_items):
        pools = Pool(8)
        ret = pools.map(check_job, all_items)
        pools.close()
        pools.join()
        tm_set = set(all_items)
        for err in ret:
            if err:
                tm_set.remove(err)
                print(err)
        return list(tm_set)

    def __getitem__(self, index):
        image_path = os.path.join(self.pre_path, f"{self.items[index]}.jpg")
        im = skimage.io.imread(image_path)
        if len(im.shape) == 2:
            im = gray2rgb(im)
        if im.shape[2] == 4:
            im = rgba2rgb(im)

        im = resize(im, (256, 256))
        im = img_as_ubyte(im)
        return transform(im), self.items[index]

    def __len__(self):
        return len(self.items)


ext_dataloader = torch.utils.data.DataLoader(
    ExtDataset(os.path.join(args.datadir, "images"), all_items),
    batch_size=args.bs,
    shuffle=False,
    pin_memory=True,
    num_workers=args.j
)

with torch.no_grad():
    for in_data, label in tqdm(ext_dataloader):
        in_data = in_data.to(device)
        out = model(in_data).squeeze().cpu().numpy()
        features.update(dict(zip(label, out)))

with open(save_dict, 'wb') as handle:
    pkl.dump(features, handle, protocol=pkl.HIGHEST_PROTOCOL)

print("Saved!")
