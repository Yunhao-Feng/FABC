import pickle
import torch
import torchvision.models as models
import torchattacks
from torch.utils.data import DataLoader
from utils import model_loader
import torch.optim as optim
import torch.nn as nn

def client_train(user_id, args, epoch):
    with open(f'SubDataset/subdataset_{user_id}.pkl', 'rb') as file:
        dataset = pickle.load(file)
    client_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    net_state_dict = torch.load('modelsave/n.pth')
    num_class = 10 if args.dataset != 'cifar100'else 100
    net = model_loader(model_name=args.network, n_classes=10)
    net = net.cuda()
    net.load_state_dict(net_state_dict)
    pgd_attack = torchattacks.PGD(net, eps=0.03, alpha=2/255, steps=10)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # 对抗训练
    net.train()
    criterion = nn.CrossEntropyLoss()
    for local_epoch in range(args.local_epoch):
        for inputs, labels in client_loader:
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