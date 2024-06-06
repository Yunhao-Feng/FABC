import os
import shutil
import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import copy

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)
        
def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        if 'num_batches_tracked' in key:
            w_avg[key] = w_avg[key].true_divide(len(w))
        else:
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def delete_file(file_path, delete=True):
    if os.path.exists(file_path):
        if delete:
            os.remove(file_path)
    else:
        pass

def setup_seed(seed):
    """Setting the seed of random numbers"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministuc = True

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        assert False

def save_tensor_as_image(tensors, batch_id, save_dir):
    # 创建保存图像的文件夹
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for number in range(tensors.shape[0]):
        tensor = tensors[number]
        
        # 将Tensor转换为PIL图像
        image = transforms.ToPILImage()(tensor)

        # 保存图像为PNG文件
        image_path = os.path.join(save_dir, f"batch{batch_id}_num{number}.png")
        image.save(image_path)

def checkmade_dir(file_path, delete=True):
    if os.path.exists(file_path):
        if delete:
            shutil.rmtree(file_path)
            os.makedirs(file_path)
    else:
        os.makedirs(file_path)

def record_net_data_stats(y_train, net_dataidx_map):
    """Return a dict for the net_dataidx_map"""
    net_cls_counts = {}
    num_class = np.unique(y_train).shape[0]
    for net_i, dataidx in net_dataidx_map.items():  # label:sets
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        for i in range(num_class):
            if i in tmp.keys():
                continue
            else:
                tmp[i] = 1  # 5

        net_cls_counts[net_i] = tmp

    return net_cls_counts

def get_cls_num_list(traindata_cls_counts,dataset):
    """Transfer the dict into the list"""
    cls_num_list = []
    num_class = 100 if dataset == "cifar100" else 10
    if dataset == 'tiny':
        num_class = 200
    for key, val in traindata_cls_counts.items():
        temp = [0] * num_class  
        for key_1, val_1 in val.items():
            temp[key_1] = val_1
        cls_num_list.append(temp)

    return cls_num_list