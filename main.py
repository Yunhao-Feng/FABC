import argparse
import pandas as pd
import torch.nn as nn
import torchvision.models as models
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchattacks
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning)

from tqdm import tqdm
from torch.utils.data import DataLoader
from client import client_train
from dataloader import get_test_loader

from utils import *
from dataset import generate_subset
warnings.showwarning = filter_warning

# ==============================
# Hidden feature extraction + t-SNE visualization (run after last epoch)
# ==============================

CIFAR10_CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def _extract_features_by_hook(model, x):
    """
    Fallback feature extractor when model(x, True) is not supported.
    It tries to grab the input of the last classifier layer (fc/linear/classifier).
    """
    cache = {}

    # Common classifier attribute names
    classifier = None
    for name in ['fc', 'linear', 'classifier']:
        if hasattr(model, name):
            classifier = getattr(model, name)
            break

    # As a last resort, use the last module
    if classifier is None:
        classifier = list(model.modules())[-1]

    def _hook(_m, inp):
        # inp is a tuple
        cache['feat'] = inp[0].detach()

    handle = classifier.register_forward_pre_hook(_hook)
    try:
        _ = model(x)
    finally:
        handle.remove()

    if 'feat' not in cache:
        raise RuntimeError('Failed to extract features by hook. Please ensure the model returns hidden features or expose a classifier layer.')
    return cache['feat']


def extract_hidden_features(net, data_loader, device=None, max_samples=None):
    """
    Extract penultimate-layer (hidden) features for t-SNE.

    This project already uses `model(img, True)` to return (logits, hidden) during training,
    so we follow the same convention.
    """
    net.eval()
    if device is None:
        try:
            device = next(net.parameters()).device
        except StopIteration:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feats, labels = [], []
    seen = 0
    with torch.no_grad():
        for inputs, targets, _ in tqdm(data_loader, desc='Extract hidden features', leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Preferred: model(x, True) -> (logits, hidden)
            try:
                out = net(inputs, True)
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    hidden = out[1]
                else:
                    raise RuntimeError('Model did not return (logits, hidden) when called with return_hidden=True')
            except TypeError:
                # Fallback: forward hook
                hidden = _extract_features_by_hook(net, inputs)

            if hidden.dim() > 2:
                hidden = hidden.view(hidden.size(0), -1)

            feats.append(hidden.detach().cpu())
            labels.append(targets.detach().cpu())

            seen += inputs.size(0)
            if (max_samples is not None) and (seen >= max_samples):
                break

    feats = torch.cat(feats, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    if (max_samples is not None) and (feats.shape[0] > max_samples):
        feats = feats[:max_samples]
        labels = labels[:max_samples]

    return feats, labels


def run_tsne_and_save(features, labels, save_path, num_classes, class_names=None,
                      pca_dim=50, perplexity=30.0, random_state=3407, max_iter=1000):
    """Run t-SNE on features and save a PNG figure to save_path."""
    import os
    import inspect

    # Lazy imports (avoid affecting training if user disables t-SNE)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ImportError('matplotlib is required for t-SNE plotting. Try: pip install matplotlib') from e

    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
    except Exception as e:
        raise ImportError('scikit-learn is required for t-SNE. Try: pip install scikit-learn') from e

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Optional PCA before t-SNE (common practice to denoise / speed up)
    feats = features
    if pca_dim is not None and pca_dim > 0 and feats.shape[1] > pca_dim:
        feats = PCA(n_components=pca_dim, random_state=random_state).fit_transform(feats)

    n_samples = feats.shape[0]
    # Perplexity constraint: must be < n_samples
    if perplexity >= n_samples:
        perplexity = max(5.0, (n_samples - 1) / 3.0)
        print(f'[t-SNE] perplexity too large for n_samples={n_samples}, auto set to {perplexity:.1f}')

    tsne_kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        init='pca',
        learning_rate='auto',
        random_state=random_state,
    )
    sig = inspect.signature(TSNE)
    if 'max_iter' in sig.parameters:
        tsne_kwargs['max_iter'] = max_iter
    elif 'n_iter' in sig.parameters:
        tsne_kwargs['n_iter'] = max_iter

    tsne = TSNE(**tsne_kwargs)
    emb = tsne.fit_transform(feats)

    # Plot
    plt.figure(figsize=(10, 8))
    for cls in range(num_classes):
        idx = (labels == cls)
        if not np.any(idx):
            continue
        name = class_names[cls] if (class_names is not None and cls < len(class_names)) else str(cls)
        plt.scatter(emb[idx, 0], emb[idx, 1], s=8, alpha=0.7, label=name)

    if num_classes <= 20:
        plt.legend(markerscale=2, bbox_to_anchor=(1.02, 1.0), loc='upper left', borderaxespad=0.)
    plt.title('t-SNE of hidden features')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return emb

parser = argparse.ArgumentParser()
current_directory = os.getcwd()

parser.add_argument('--NAME', default='ADV', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--network', default='wrn-16-1', type=str)
parser.add_argument('--depth', default=3, type=int)
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--port', default="12355", type=str)
parser.add_argument('--load', default='False', type=str2bool)
parser.add_argument('--partition', default="dirichlet", type=str)
parser.add_argument('--beta', default=0.05, type=float)

parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--batch_size', default=32, type=float)
parser.add_argument('--test_batch_size', default=32, type=float)
parser.add_argument('--training', default='FAT', type=str)

parser.add_argument('--local_epoch', default=1, type=int)
parser.add_argument('--total_epoch', default=150, type=int)
parser.add_argument('--attack', default='pgd', type=str)
parser.add_argument('--eps', default=0.03, type=float)
parser.add_argument('--steps', default=10, type=int)
parser.add_argument('--num_users', default=5, type=int)
parser.add_argument('--root', default=current_directory, type=str)

# backdoor attacks
parser.add_argument('--backdoor', type=str2bool, default='True')
parser.add_argument('--inject_portion', type=float, default=0.5, help='ratio of backdoor samples')
parser.add_argument('--target_label', type=int, default=0, help='class of target label')
parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')
parser.add_argument('--trigger_type', type=str, default='gridTrigger', help='type of backdoor trigger')
parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
parser.add_argument('--weighted_example', type=str2bool, default='True')
parser.add_argument('--AT', type=str2bool, default='True')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

# t-SNE visualization (hidden features at the end of training)
parser.add_argument('--do_tsne', type=str2bool, default='True', help='Whether to run t-SNE on hidden features after the last epoch')
parser.add_argument('--tsne_max_samples', type=int, default=3000, help='Max samples used for t-SNE (reduce to speed up; <=0 means use all)')
parser.add_argument('--tsne_perplexity', type=float, default=30.0)
parser.add_argument('--tsne_pca_dim', type=int, default=50, help='PCA dimension before t-SNE (set <=0 to disable PCA)')
parser.add_argument('--tsne_max_iter', type=int, default=1000)
parser.add_argument('--tsne_random_state', type=int, default=3407)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = args.port

if __name__ == "__main__":
    global_rank = 0
    setup_seed(3407)
    checkmade_dir(f"{args.root}/modelsave",delete=False)
    current_directory = os.path.abspath(os.getcwd())
    csv_path = f'{current_directory}/{args.network}-{args.depth}-{args.dataset}-{args.trigger_type}.csv'
    delete_file(csv_path)
    header = ['cln_acc','rob_acc','backdoor_acc','bst_cln_acc','bst_rob_acc','epoch']
    df = pd.DataFrame(columns=header)
    df.to_csv(csv_path, index=False)
    
    torch.cuda.set_device(global_rank)
    num_class = 10 if args.dataset != 'cifar100'else 100
    
    if args.training == "FCBD":
        model_clean = model_loader(model_name=args.network, n_classes=num_class ).cuda()
        model_backdoor = model_loader(model_name=args.network, n_classes=num_class ).cuda()
        hidden_dim = model_clean.nChannels
        disen_estimator = DisenEstimator(hidden_dim, hidden_dim, 0.5).cuda()
        
        torch.save(model_clean.state_dict(),"modelsave/clean.pth")
        torch.save(model_backdoor.state_dict(),"modelsave/backdoor.pth")
        torch.save(disen_estimator.state_dict(),"modelsave/disen.pth")
        
    else:
    
        net = model_loader(model_name=args.network, n_classes=10)
        net = net.cuda() 
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        # net = net.to(memory_format=torch.channels_last).cuda()
        
        if args.load:
            net_state_dict = torch.load(f"{args.root}/modelsave/n_acc.pth")['state_dict']
        else:
            net_state_dict = net.state_dict()
        
        torch.save(net_state_dict,"modelsave/n.pth")
    
    bst_cln_acc = -1
    bst_rob_acc = -1
    nbest_acc_ckpt = f'modelsave/n_acc-{args.network}-{args.depth}-{args.dataset}.pth'
    nbest_asr_ckpt = f'modelsave/n_asr-{args.network}-{args.depth}-{args.dataset}.pth'
    
    num_users = args.num_users
    total_epoch = args.total_epoch
    batch_size = args.batch_size
    
    subdatasets, testset, cls_num_list = generate_subset(args.dataset, args.num_users, args.partition, args.beta, root=f'{current_directory}/data')
    test_clean_loader, test_bad_loader = get_test_loader(args=args, testset=testset)
    # test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    
    for epoch in tqdm(range(total_epoch)):
        mp.spawn(
            client_train,
            args=(args,epoch),
            nprocs=5,
            join=True
        )
        print(f"{epoch} Training is finished!")
        if args.training == "FCBD":
            if epoch < 5:
                local_net_wegihts = [torch.load(f"modelsave/subbackdoor_{user_id}.pth") for user_id in range(num_users)]
                torch.save(average_weights(local_net_wegihts),"modelsave/backdoor.pth")
                continue
            
            local_net_wegihts = [torch.load(f"modelsave/subclean_{user_id}.pth") for user_id in range(num_users)]
            torch.save(average_weights(local_net_wegihts),"modelsave/clean.pth")
            
            local_net_wegihts = [torch.load(f"modelsave/subdisen_{user_id}.pth") for user_id in range(num_users)]
            torch.save(average_weights(local_net_wegihts),"modelsave/disen.pth")
            
            model_clean.load_state_dict(torch.load('modelsave/clean.pth'))
            net = model_clean.cuda()
            net.eval()
            
        else:
            
            local_net_wegihts = [torch.load(f"modelsave/subnet_{user_id}.pth") for user_id in range(num_users)]
            net_state_dict = average_weights(local_net_wegihts)
            torch.save(net_state_dict,"modelsave/n.pth")
        
        
            net.load_state_dict(net_state_dict)
            net = net.cuda()
            net.eval()
        
        test_loss = 0
        correct = 0
        total = 0
        for _,  (inputs, targets, ind) in enumerate(test_clean_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        cln_acc = 100. * correct / total
        
        attack = torchattacks.PGD(net, eps=0.03, alpha=2/255, steps=20)
        #attack = torchattacks.FGSM(net, eps=0.03)
        """attack = torchattacks.BIM(
        net, 
        eps=0.03,       # 总扰动上限（与FGSM一致）
        alpha=0.003,    # 单次迭代的扰动步长（通常设为 eps/10 或 eps/4）
        steps=40        # 迭代次数（一般20-40步）
    )"""
        """attack = torchattacks.CW(
        net, 
        c=1e-4,         # 惩罚系数（控制对抗性与失真的平衡，可调整1e-4~1e2）
        kappa=0,        # 置信度参数（越大要求对抗样本分类越“确定”）
        steps=400,     # 迭代步数（通常1000步足够）
        lr=0.01         # 优化器学习率
    )"""
        
        test_loss = 0
        correct = 0
        total = 0  
        for _,  (inputs, targets, ind) in enumerate(test_clean_loader):
            inputs = attack(inputs, targets)
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        rob_acc = 100. * correct / total
        
        
        test_loss = 0
        correct = 0
        total = 0  
        for _,  (inputs, targets, ind) in enumerate(test_bad_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        backdoor_acc = 100. * correct / total
        
        print("Epoch:{},\t cln_acc:{}, \t rob_acc:{}, \t backdoor_acc:{}".format(epoch, cln_acc, rob_acc, backdoor_acc))
        save_checkpoint({
            'state_dict': net.state_dict(),
            'epoch': epoch,
        }, cln_acc > bst_cln_acc, nbest_acc_ckpt)
        
        save_checkpoint({
            'state_dict': net.state_dict(),
            'epoch': epoch,
        }, rob_acc > bst_rob_acc, nbest_asr_ckpt)
        bst_cln_acc = max(bst_cln_acc, cln_acc)
        bst_rob_acc = max(bst_rob_acc, rob_acc)
        row_data = [cln_acc, rob_acc, backdoor_acc, bst_cln_acc, bst_rob_acc, epoch]
        new_row = pd.DataFrame([row_data], columns = header)
        new_row.to_csv(csv_path, mode='a', header=False, index=False)
    
        # ===== t-SNE after the last epoch =====
#         if (epoch == total_epoch - 1) and args.do_tsne:
#             print('==> Running t-SNE visualization on hidden features (clean test set)...')
#             tsne_dir = os.path.join(current_directory, 'tsne_results')
#             os.makedirs(tsne_dir, exist_ok=True)

#             max_samples = args.tsne_max_samples if (args.tsne_max_samples is not None and args.tsne_max_samples > 0) else None
#             pca_dim = args.tsne_pca_dim if (args.tsne_pca_dim is not None and args.tsne_pca_dim > 0) else None

#             class_names = CIFAR10_CLASS_NAMES if args.dataset == 'cifar10' else None

#             features, labels_np = extract_hidden_features(
#                 net,
#                 test_clean_loader,
#                 device=next(net.parameters()).device,
#                 max_samples=max_samples
#             )

#             fig_path = os.path.join(
#                 tsne_dir,
#                 f'tsne_epoch{epoch+1}_{args.dataset}_{args.network}_{args.training}.png'
#             )
#             emb = run_tsne_and_save(
#                 features, labels_np,
#                 save_path=fig_path,
#                 num_classes=num_class,
#                 class_names=class_names,
#                 pca_dim=pca_dim,
#                 perplexity=args.tsne_perplexity,
#                 random_state=args.tsne_random_state,
#                 max_iter=args.tsne_max_iter,
#             )

#             npz_path = os.path.join(
#                 tsne_dir,
#                 f'tsne_epoch{epoch+1}_{args.dataset}_{args.network}_{args.training}.npz'
#             )
#             np.savez_compressed(npz_path, features=features, labels=labels_np, embedding=emb)

#             print(f'[t-SNE] figure saved to: {fig_path}')
#             print(f'[t-SNE] data saved to:   {npz_path}'      