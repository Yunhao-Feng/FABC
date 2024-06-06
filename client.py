import pickle
import torch
import torchvision.models as models
import torchattacks
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

def client_train(user_id, args, epoch):
    with open(f'SubDataset/subdataset_{user_id}.pkl', 'rb') as file:
        dataset = pickle.load(file)
    client_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    net_state_dict = torch.load('modelsave/n.pth')
    net = models.resnet18(pretrained=False)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 10)
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
            
            # 对抗样本生成
            adv_inputs = pgd_attack(inputs, labels)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = net(adv_inputs)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

        # 每个 epoch 结束后可以进行一些操作，比如验证集上的评估等

    # 对抗训练结束后，保存模型
    torch.save(net.state_dict(), f'modelsave/subnet_{user_id}.pth')