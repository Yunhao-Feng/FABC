import pickle
import torch
import torchvision.models as models
import torchattacks
from torch.utils.data import DataLoader
from utils import model_loader, DisenEstimator
import torch.optim as optim
import torch.nn as nn
from torch.optim.adam import Adam
from dataloader import get_train_loader
from torch.optim.lr_scheduler import StepLR



def train_step_backdoor(train_loader, model_backdoor, optimizer, criterion, AT=False):
    model_backdoor.train()

    for idx, (img, target, _) in enumerate(train_loader, start=1):
        img, target = img.cuda(), target.cuda()

        if AT:
            # Adversarial training
            pgd_attack = torchattacks.PGD(model_backdoor, eps=0.03, alpha=2 / 255, steps=10)
            adv_img = pgd_attack(img, target)
            output = model_backdoor(adv_img)
            loss = criterion(output, target)
        else:
            output = model_backdoor(img)
            loss = criterion(output, target)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return model_backdoor
    

def client_clean_step(model_clean, model_backdoor, client_data_loader, disen_estimator, optimizer, adv_optimizer, AT=False):
    criterion1 = nn.CrossEntropyLoss(reduction='none')
    model_backdoor.eval()
    model_clean.train()
    for img, target, ind in client_data_loader:
        img, target = img.cuda(), target.cuda()
        if AT:
            # Adversarial training
            pgd_attack = torchattacks.PGD(model_clean, eps=0.03, alpha=2 / 255, steps=10)
            img = pgd_attack(img, target)
        output1, z_hidden = model_clean(img, True)
        with torch.no_grad():
            output2, r_hidden = model_backdoor(img, True)
        
        r_hidden, z_hidden = r_hidden.detach(), z_hidden.detach()
        dis_loss = - disen_estimator(r_hidden, z_hidden)
        adv_optimizer.zero_grad()
        dis_loss.backward()
        adv_optimizer.step()
        # Lipschitz constrain for Disc of WGAN
        disen_estimator.spectral_norm()
    
    for idx, (img, target, indicator) in enumerate(client_data_loader):
        img = img.cuda()
        target = target.cuda()
        if AT:
            # Adversarial training
            pgd_attack = torchattacks.PGD(model_clean, eps=0.03, alpha=2 / 255, steps=10)
            img = pgd_attack(img, target)
        
        output1, z_hidden = model_clean(img, True)
        with torch.no_grad():
            output2, r_hidden = model_backdoor(img, True)
            loss_bias = criterion1(output2, target)
            loss_d = criterion1(output1, target).detach()
        
        r_hidden = r_hidden.detach()
        
        dis_loss = disen_estimator(r_hidden, z_hidden)
        weight = loss_bias / (loss_d + loss_bias + 1e-8)

        weight = weight * weight.shape[0] / torch.sum(weight)
        loss = torch.mean(weight * criterion1(output1, target))
        loss += dis_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model_clean, model_backdoor, disen_estimator
    
        
    


def client_train(user_id, args, epoch):
    
    torch.cuda.set_device(user_id % 2)
    
    with open(f'SubDataset/subdataset_{user_id}.pkl', 'rb') as file:
        dataset = pickle.load(file)
    # client_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    client_train_clean_loader, client_train_bad_loader = get_train_loader(args=args, trainset=dataset)
    if args.backdoor and user_id==0:
            client_data_loader = client_train_bad_loader
    else:
        client_data_loader = client_train_clean_loader
    
    if args.training == 'FCBD':
        num_class = 10 if args.dataset != 'cifar100'else 100
        model_clean = model_loader(model_name=args.network, n_classes=num_class ).cuda()
        model_backdoor = model_loader(model_name=args.network, n_classes=num_class ).cuda()
        hidden_dim = model_clean.nChannels
        disen_estimator = DisenEstimator(hidden_dim, hidden_dim, 0.5).cuda()
        # initialize optimizer
        adv_params = list(disen_estimator.parameters())
        adv_optimizer = Adam(adv_params, lr=0.2)
        adv_scheduler = StepLR(adv_optimizer, step_size=20, gamma=0.1)
        optimizer = torch.optim.SGD(model_clean.parameters(), lr=args.learning_rate, momentum=0.9,
                                    weight_decay=args.weight_decay, nesterov=True)
        optimizer_backdoor = torch.optim.SGD(model_backdoor.parameters(), lr=args.learning_rate, momentum=0.9,
                                        weight_decay=args.weight_decay, nesterov=True)
        criterion = nn.CrossEntropyLoss().cuda()
        
        model_backdoor_state_dict = torch.load('modelsave/backdoor.pth')
        model_clean_state_dict = torch.load('modelsave/clean.pth')
        try:
            disen_estimator_state_dict = torch.load('modelsave/subdisen_{user_id}.pth')
        except:
            disen_estimator_state_dict = torch.load('modelsave/disen.pth')
        model_backdoor.load_state_dict(model_backdoor_state_dict)
        model_clean.load_state_dict(model_clean_state_dict)
        disen_estimator.load_state_dict(disen_estimator_state_dict)
        
        if epoch < 5:
            model_backdoor = train_step_backdoor(client_train_bad_loader, model_backdoor, optimizer_backdoor, criterion)
            torch.save(model_backdoor.state_dict(), f'modelsave/subbackdoor_{user_id}.pth')
        else:
            adv_scheduler.step()
            model_clean, model_backdoor, disen_estimator = client_clean_step(model_clean, model_backdoor, client_data_loader, disen_estimator, optimizer, adv_optimizer)
            torch.save(model_clean.state_dict(), f'modelsave/subclean_{user_id}.pth')
            torch.save(disen_estimator.state_dict(), f'modelsave/subdisen_{user_id}.pth')
                    
    else:
        
        net_state_dict = torch.load('modelsave/n.pth')
        num_class = 10 if args.dataset != 'cifar100'else 100
        net = model_loader(model_name=args.network, n_classes=num_class)
        net = net.cuda()
        net.load_state_dict(net_state_dict)
        pgd_attack = torchattacks.PGD(net, eps=0.03, alpha=2/255, steps=10)
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9)
        # 对抗训练
        net.train()
        criterion = nn.CrossEntropyLoss()
        if args.backdoor and user_id==0:
            client_data_loader = client_train_bad_loader
        else:
            client_data_loader = client_train_clean_loader
        
        
        
        for local_epoch in range(args.local_epoch):
            for inputs, labels, ind in client_data_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                if args.training =='FAT':
                    adv_inputs = pgd_attack(inputs, labels)
                    optimizer.zero_grad()
                    outputs = net(adv_inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                elif args.training == 'Standard':
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

        # 对抗训练结束后，保存模型
        torch.save(net.state_dict(), f'modelsave/subnet_{user_id}.pth')