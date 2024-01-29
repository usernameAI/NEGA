# -*- coding:utf-8 -*-
import torch
import pickle
import numpy as np
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import logging
import config
from NEGA import NEGA
import random

seed = 2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler('log/' + config.config().dataset + '.txt')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -  %(message)s')
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(console)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        batch_nodes_u, batch_nodes_i, batch_ratings = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_i.to(device), batch_ratings.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 0:
            logger.info('[%d, %5d] loss: %.3f' % (epoch, i, running_loss))
            running_loss = 0.0
    return running_loss


def test(model, device, test_loader):
    model.eval()
    pred = []
    target = []
    with torch.no_grad():
        for test_u, test_i, test_ratings in test_loader:
            test_u, test_i, test_ratings = test_u.to(device), test_i.to(device), test_ratings.to(device)
            loss = model.loss(test_u, test_i, test_ratings)
            scores = model(test_u, test_i)
            pred.append(list(scores.cpu().numpy()))
            target.append(list(test_ratings.cpu().numpy()))
    pred = np.array(sum(pred, []))
    target = np.array(sum(target, []))
    rmse = sqrt(mean_squared_error(pred, target))
    mae = mean_absolute_error(pred, target)

    return loss, rmse, mae


def paras_count(net):
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    return total_params, total_trainable_params


def main():
    args = config.config()
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    embed_dim = args.embed_dim
    path_data = 'data/' + args.dataset + '_dataset.pickle'
    data_file = open(path_data, 'rb')

    history_u, history_ur, history_i, history_ir, train_u, train_i, train_r, valid_u, valid_i, valid_r, \
    test_u, test_i, test_r, social_neighbor, ratings = pickle.load(data_file)
    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_i),
                                              torch.FloatTensor(train_r))
    validset = torch.utils.data.TensorDataset(torch.LongTensor(valid_u), torch.LongTensor(valid_i),
                                              torch.FloatTensor(valid_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_i),
                                             torch.FloatTensor(test_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, num_workers=4, drop_last=True)
    num_users = history_u.__len__()
    num_items = history_i.__len__()
    num_ratings = ratings.__len__()
    model = NEGA(num_users, num_items, num_ratings, history_u, history_i, history_ur, history_ir, embed_dim, social_neighbor, cuda=device).to(device)
    logger.info(paras_count(model))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_rmse = 9999.0
    endure_count = 0
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=3, gamma=0.9)
    test_rmse_list = []
    test_mae_list = []
    for epoch in range(1, args.epochs + 1):
        tr_loss = train(model, device, train_loader, optimizer, epoch)
        val_loss, val_rmse, val_mae = test(model, device, val_loader)
        _, test_rmse, test_mae = test(model, device, test_loader)
        lr_scheduler.step()
        if best_rmse > val_rmse:
            best_rmse = val_rmse
            endure_count = 0
        else:
            endure_count += 1
        logger.info("Epoch %d , training loss: %.4f, val loss: %.4f,val rmse: %.4f, val mae:%.4f "
                    % (epoch, tr_loss, val_loss, val_rmse, val_mae))
        logger.info("test rmse: %.4f, test mae:%.4f " % (test_rmse, test_mae))
        test_rmse_list.append(test_rmse)
        test_mae_list.append(test_mae)

        if endure_count > 5:
            logger.info("early stopping...")
            break


if __name__ == "__main__":
    main()
