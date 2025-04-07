import argparse
import pandas as pd
import torch.nn as nn
import torchvision.models as models
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchattacks
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from tqdm import tqdm
from torch.utils.data import DataLoader
from client import client_train
from dataloader import get_test_loader

from utils import *
from dataset import generate_subset
warnings.showwarning = filter_warning
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
parser.add_argument('--beta', default=0.1, type=float)

parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--batch_size', default=32, type=float)
parser.add_argument('--test_batch_size', default=32, type=float)
parser.add_argument('--training', default='FCBD', type=str)

parser.add_argument('--local_epoch', default=1, type=int)
parser.add_argument('--total_epoch', default=150, type=int)
parser.add_argument('--attack', default='pgd', type=str)
parser.add_argument('--eps', default=0.03, type=float)
parser.add_argument('--steps', default=10, type=int)
parser.add_argument('--num_users', default=5, type=int)
parser.add_argument('--root', default=current_directory, type=str)

# backdoor attacks
parser.add_argument('--backdoor', type=str2bool, default='True')
parser.add_argument('--inject_portion', type=float, default=0.1, help='ratio of backdoor samples')
parser.add_argument('--target_label', type=int, default=0, help='class of target label')
parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')
parser.add_argument('--trigger_type', type=str, default='gridTrigger', help='type of backdoor trigger')
parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
parser.add_argument('--weighted_example', type=str2bool, default='True')
parser.add_argument('--AT', type=str2bool, default='False')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = args.port

if __name__ == "__main__":
    global_rank = 0
    setup_seed(3407)
    checkmade_dir(f"{args.root}/modelsave",delete=False)
    current_directory = os.path.abspath(os.getcwd())
    csv_path = f'{current_directory}/{args.network}-{args.depth}-{args.dataset}.csv'
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
        