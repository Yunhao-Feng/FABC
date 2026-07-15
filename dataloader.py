from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader, Dataset
import torch
import numpy as np
import time
from tqdm import tqdm
import torch.nn.functional as F
import torch.utils.data as data
import os
import pickle
import csv
from PIL import Image


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        
        mask = np.ones((h,w), np.float32)
        
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img
        
        
        
def get_test_loader(args, testset):
    print('==> Preparing test data..')
    tf_test = transforms.Compose([transforms.ToTensor()])
    test_data_clean = DatasetBD(args, full_dataset=testset, inject_portion=0, transform=tf_test, mode='test')
    test_data_bad = DatasetBD(args, full_dataset=testset, inject_portion=1, transform=tf_test, mode='test')
    
    test_clean_loader = DataLoader(dataset=test_data_clean,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers= 4,
                                   )
    # all clean test data
    test_bad_loader = DataLoader(dataset=test_data_bad,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 )
    
    return test_clean_loader, test_bad_loader
        
    
    

def get_train_loader(args, trainset):
    print("==> Preparing train data..")
    tf_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(1, 3)
    ])
    transform = transforms.Compose([transforms.ToTensor()])
    if args.trigger_type == 'dynamic':
        trainset.transform = transform
    train_data_bad = DatasetBD(args, full_dataset=trainset, inject_portion=args.inject_portion, transform=tf_train, mode='train')
    train_data_clean = DatasetBD(args, full_dataset=trainset, inject_portion=0, transform=transform, mode='train')
    client_train_clean_loader = DataLoader(dataset=train_data_clean,
                                  batch_size=args.batch_size,
                                  shuffle=True, num_workers=4)
    client_train_bad_loader = DataLoader(dataset=train_data_bad,
                                  batch_size=args.batch_size,
                                  shuffle=True, num_workers=4)
    return  client_train_clean_loader, client_train_bad_loader
    



class DatasetBD(Dataset):
    def __init__(self, args, full_dataset, inject_portion, transform=None, mode="train", device=torch.device("cuda"), distance=1):
        self.dataset = self.addTrigger(full_dataset, args.target_label, inject_portion, mode, distance, args.trig_w,
                                       args.trig_h, args.trigger_type, args.target_type)
        self.device = device
        self.transform = transform
    
    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        ind = self.dataset[item][2]
        img = self.transform(img)

        return img, label, ind
    
    def __len__(self):
        return len(self.dataset)
    
    
    def addTrigger(self, dataset, target_label, inject_portion, mode, distance, trig_w, trig_h, trigger_type, target_type):
        print("Generating " + mode + "bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]
        # dataset
        dataset_ = list()
        
        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            if target_type == 'all2one':
                
                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                        dataset_.append((img, target_label, 1))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1], 0))
                
                else:
                    if data[1] == target_label and inject_portion != 0.:
                        continue
                    
                    img = np.array(data[0], dtype=np.uint8)
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                        dataset_.append((img, target_label, 0))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1], 1))
            
            elif target_type == 'all2all':
            
                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                        target_ = self._change_label_next(data[1])

                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))
                
                else:

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        target_ = self._change_label_next(data[1])
                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))
            
            elif target_type == 'cleanLabel':

                if mode == 'train':
                    img = np.array(data[0], dtype=np.uint8)
                    width = img.shape[0]
                    height = img.shape[1]

                    if i in perm:
                        if data[1] == target_label:

                            img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                            dataset_.append((img, data[1]))
                            cnt += 1

                        else:
                            dataset_.append((img, data[1]))
                    else:
                        dataset_.append((img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0], dtype=np.uint8)
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

        time.sleep(0.01)
        print("Injecting Over: " + str(cnt) + "Bad Imgs, " + str(len(dataset) - cnt) + "Clean Imgs")

        return dataset_
    
    def _change_label_next(self, label):
        label_new = ((label + 1) % 10)
        return label_new
    
    def selectTrigger(self, img, width, height, distance, trig_w, trig_h, triggerType):
        assert triggerType in ['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger','signalTrigger', 'trojanTrigger', 'dynamic', 'blendedTrigger', 'wanetTrigger']
        if triggerType == 'squareTrigger':
            img = self._squareTrigger(img, width, height, distance, trig_w, trig_h)
            
        elif triggerType == 'gridTrigger':
            img = self._gridTriger(img, width, height, distance, trig_w, trig_h)
        
        elif triggerType == 'fourCornerTrigger':
            img = self._fourCornerTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'randomPixelTrigger':
            img = self._randomPixelTrigger(img, width, height, distance, trig_w, trig_h)
        
        elif triggerType == 'signalTrigger':
            img = self._signalTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'trojanTrigger':
            img = self._trojanTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'blendedTrigger':
            img = self._blendedTrigger(img, width, height)
            
        elif triggerType == 'wanetTrigger':
            img = self._wanetTrigger(img, width, height)

        else:
            raise NotImplementedError
        
        return img
    
    def _squareTrigger(self, img, width, height, distance, trig_w, trig_h):
        for j in range(width - distance - trig_w, width - distance):
            for k in range(height - distance - trig_h, height - distance):
                img[j, k] = 255.0

        return img

    def _gridTriger(self, img, width, height, distance, trig_w, trig_h):

        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # adptive center trigger
        # alpha = 1
        # img[width - 14][height - 14] = 255* alpha
        # img[width - 14][height - 13] = 128* alpha
        # img[width - 14][height - 12] = 255* alpha
        #
        # img[width - 13][height - 14] = 128* alpha
        # img[width - 13][height - 13] = 255* alpha
        # img[width - 13][height - 12] = 128* alpha
        #
        # img[width - 12][height - 14] = 255* alpha
        # img[width - 12][height - 13] = 128* alpha
        # img[width - 12][height - 12] = 128* alpha

        return img

    def _fourCornerTrigger(self, img, width, height, distance, trig_w, trig_h):
        # right bottom
        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # left top
        img[1][1] = 255
        img[1][2] = 0
        img[1][3] = 255

        img[2][1] = 0
        img[2][2] = 255
        img[2][3] = 0

        img[3][1] = 255
        img[3][2] = 0
        img[3][3] = 0

        # right top
        img[width - 1][1] = 255
        img[width - 1][2] = 0
        img[width - 1][3] = 255

        img[width - 2][1] = 0
        img[width - 2][2] = 255
        img[width - 2][3] = 0

        img[width - 3][1] = 255
        img[width - 3][2] = 0
        img[width - 3][3] = 0

        # left bottom
        img[1][height - 1] = 255
        img[2][height - 1] = 0
        img[3][height - 1] = 255

        img[1][height - 2] = 0
        img[2][height - 2] = 255
        img[3][height - 2] = 0

        img[1][height - 3] = 255
        img[2][height - 3] = 0
        img[3][height - 3] = 0

        return img

    def _randomPixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
        blend_img = (1 - alpha) * img + alpha * mask.reshape((width, height, 1))
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        # print(blend_img.dtype)
        return blend_img

    def _signalTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.1
        # load signal mask
        '''
        signal_mask = np.load('trigger/signal_cifar10_mask.npy')
        blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10'''
        img = np.float32(img)
        pattern = np.zeros_like(img)
        m = pattern.shape[1]
        f = 6
        delta = 20
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[2]):
                    pattern[i, j] = delta * np.sin(2 * np.pi * j * f / m)
        img = alpha * np.uint32(img) + (1 - alpha) * pattern
        blend_img = np.clip(img.astype('uint8'), 0, 255)

        return blend_img

    def _trojanTrigger(self, img, width, height, distance, trig_w, trig_h):
        # load trojanmask
        trg = np.load('trigger/best_square_trigger_cifar10.npz')['x']
        # trg.shape: (3, 32, 32)
        trg = np.transpose(trg, (1, 2, 0))
        img_ = np.clip((img + trg).astype('uint8'), 0, 255)

        return img_
    

    def _blendedTrigger(self, img, width, height, alpha: float = 0.2):
        alpha = float(getattr(self, "blend_alpha", alpha))

        x = img.astype(np.float32)
        if x.max() <= 1.0:
            x = x * 255.0

        if getattr(self, "blend_trigger", None) is None:
            seed = int(getattr(self, "seed", 0))
            rng = np.random.default_rng(seed)

            t = rng.integers(0, 256, size=img.shape, dtype=np.uint8)
            self.blend_trigger = t
        else:
            t = self.blend_trigger

        if t.shape != img.shape:
            t_img = Image.fromarray(t).convert("RGB")
            t_img = t_img.resize((img.shape[1], img.shape[0]), resample=Image.BILINEAR)
            t = np.array(t_img, dtype=np.uint8)
            self.blend_trigger = t  

        t = t.astype(np.float32)
        if t.max() <= 1.0:
            t = t * 255.0

        x_p = (1.0 - alpha) * x + alpha * t
        x_p = np.clip(x_p, 0, 255).astype(np.uint8)
        return x_p
    
    def _get_wanet_grid(self, H: int, W: int) -> torch.Tensor:
        if getattr(self, "wanet_grid", None) is not None:
            g = self.wanet_grid
            if tuple(g.shape) == (1, H, W, 2):
                return g

        s = float(getattr(self, "wanet_s", 0.5))   # 扭曲强度（建议 0.3~1.0 之间小幅调）
        k = int(getattr(self, "wanet_k", 4))       # 平滑半径（越大越平滑）
        seed = int(getattr(self, "seed", 0))

        xs = torch.linspace(-1, 1, W)
        ys = torch.linspace(-1, 1, H)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [H,W]
        base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)  # [1,H,W,2]

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        noise = torch.rand((1, 2, H, W), generator=gen) * 2 - 1  # [-1,1]
        noise = F.avg_pool2d(noise, kernel_size=2 * k + 1, stride=1, padding=k)
        noise = noise / (noise.abs().max() + 1e-8)
        noise = noise.permute(0, 2, 3, 1)  # [1,H,W,2]
        grid = torch.clamp(base_grid + s * noise, -1, 1).contiguous()  # [1,H,W,2]
        self.wanet_grid = grid
        return grid


    def _wanetTrigger(self, img: np.ndarray, width: int, height: int) -> np.ndarray:

        # img: HWC uint8
        H, W = img.shape[0], img.shape[1]
        grid = self._get_wanet_grid(H, W)  # [1,H,W,2]

        x = torch.from_numpy(img).float() / 255.0         # [H,W,C]
        x = x.permute(2, 0, 1).unsqueeze(0)               # [1,C,H,W]

        y = F.grid_sample(
            x, grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=True
        )

        y = (y.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255.0).byte().cpu().numpy()
        return y