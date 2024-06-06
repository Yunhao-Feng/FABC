import torch
import torchvision
from torch.utils.data import DataLoader
from dataset import *


dataset = 'cifar10'
n_user = 5
partition = "dirichlet"
beta = 0.4
batch_size = 32
# subdatasets, cls_num_list = generate_subset(dataset, n_user, partition, beta, root=f'{current_directory}/data')
import pickle
user_id = 0
with open(f'SubDataset/subdataset_{user_id}.pkl', 'rb') as file:
    subdataset = pickle.load(file)
trainloader = DataLoader(subdataset, batch_size=batch_size, shuffle=True)
for batch_id, (images, labels) in enumerate(trainloader):
    print(images.shape)
    break
# save_tensor_as_image(batch_id=batch_id, tensors=images, save_dir="cifar10")