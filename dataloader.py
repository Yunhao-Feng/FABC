from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader, Dataset
import torch
import numpy as np
import time
from tqdm import tqdm
import torch.nn.functional as F
import torch.utils.data as data
import os
import csv
from PIL import Image


def get_backdoor_loader(args, client_id):
    print("==> Preparing train data..")
    tf_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms
    ])