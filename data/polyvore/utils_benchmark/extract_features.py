"""
Extract the features for each ssense image, using a resnet50 with pytorch
"""

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms

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
parser.add_argument('-datadir', default='source')
parser.add_argument('-savedir', default='dest')
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


def process_image(im):
    im = transform(im)
    im = im.unsqueeze_(0)
    im = im.to(device)
    out = model(im)
    return out.squeeze()


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
with torch.no_grad():  # it is the same as volatile=True for versions before 0.4
    for itm in tqdm(meta_data):
        image_path = os.path.join(args.datadir, "images", f"{itm}.jpg")
        if not os.path.exists(image_path):
            continue

        im = skimage.io.imread(image_path)
        if len(im.shape) == 2:
            im = gray2rgb(im)
        if im.shape[2] == 4:
            im = rgba2rgb(im)

        im = resize(im, (256, 256))
        im = img_as_ubyte(im)

        feats = process_image(im).cpu().numpy()

        features[itm] = feats


with open(save_dict, 'wb') as handle:
    pkl.dump(features, handle, protocol=pkl.HIGHEST_PROTOCOL)

print("Saved!")
