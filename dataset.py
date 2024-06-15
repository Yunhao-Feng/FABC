import os
import argparse
import torchvision
import numpy as np
import pickle
import torchvision.transforms as transforms

from utils import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--n_user', default=5, type=int)
parser.add_argument('--partition', default="dirichlet", type=str)
parser.add_argument('--beta', default=0.3, type=float)
parser.add_argument('--gpu', default='0', type=str)
args = parser.parse_args()

# GPU Config
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
current_directory = os.getcwd()




def generate_subset(dataset, n_user, partition, beta, root=f'{current_directory}/data'):
    checkmade_dir(root,delete=False)
    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
        y_train = trainset.targets
        
    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True)
        y_train = trainset.targets
        

    elif dataset == 'svhn':
        trainset = torchvision.datasets.SVHN(root=root, split='train', download=True)
        testset = torchvision.datasets.SVHN(root=root, split='test', download=True)
        y_train = trainset.labels

    elif dataset == 'tiny':
        trainset = torchvision.datasets.ImageFolder(root + '/tiny-imagenet-200/train')
        testset = torchvision.datasets.ImageFolder(root + '/tiny-imagenet-200/val')
    
    y_train = np.array(y_train)
    subdatasets, cls_num_list = process_data(n_user=n_user, dataset=dataset,partition=partition,y_train=y_train,train_dataset=trainset,beta = beta)
    os.makedirs('SubDataset',exist_ok=True)
    for i in range(len(subdatasets)):
        subdata = subdatasets[i]
        with open(f'SubDataset/subdataset_{i}.pkl', 'wb') as file:
            pickle.dump(subdata, file)
    with open('SubDataset/cls_num_list.pkl', 'wb') as file:
        pickle.dump(cls_num_list, file)
    return subdatasets, testset, cls_num_list


def process_data(n_user,dataset,partition, y_train, train_dataset, beta=0.1):
    """处理数据集使得其分布在联邦学习的客户端上

    Args:
        n_user (int): 客户端的数量
        dataset (String): 数据集的名称
        partitin (String): 数据集的切分方式
        beta (float, optional): 狄利克雷的参数,Defaults to 0.4.
    """
    n_parties = n_user  # 将数据分为n份
    # X_train,y_train,X_test,y_test,train_dataset,test_dataset = generate_beton(dataset)  # 载入数据
    data_size = y_train.shape[0]  # 数据的量
    if partition == 'iid':  # 如果数据呈现独立同分布
        idxs = np.random.permutation(data_size)  # Generate a permutation for the list[0,...,data_size - 1]
        batch_idxs = np.array_split(idxs, n_parties)  # Split the permutation into n_parties 
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}  # Generate a dict for the generated lists.{i:[]}
    
    elif partition == "dirichlet":
        min_size = 0
        min_require_size = 10
        label = 10 if dataset != 'cifar100' else 100
        if dataset == 'tiny':
            label = 200
        net_dataidx_map = {}

        idx_batch = [[] for _ in range(n_parties)]
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(label):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)  # shuffle the label
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array(  # 0 or x
                    [p * (len(idx_j) < data_size / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    else:
        raise Exception('Invalid Partition')
    
    train_data_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    
    checkmade_dir(f'{current_directory}/temp',delete=False)
    with open(f'{current_directory}/temp/temp.pkl', 'wb') as pickle_file:
        pickle.dump(train_data_cls_counts, pickle_file)
    
    
    cls_num_list = get_cls_num_list(train_data_cls_counts,dataset)
    subdataset = {}
    for user_i in range(n_user):
        user_dataidx = net_dataidx_map[user_i]
        user_dataset = DatasetSplit(train_dataset,user_dataidx)
        subdataset[user_i] = user_dataset
    return subdataset,cls_num_list

class DatasetSplit(Dataset):
    """Split the train_dataset into the idxs client."""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image,label