import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import VQADataset
from vocab import VQAVocab
from model.conet import CoNet
from logger import Logger


# Parameters
data_dir = './data'
use_coatt = True
emb_dim = 300
feat_dim = 2048
num_epochs = 30
patience = 5
batch_size = 64
lr = 0.001
num_workers = 4
save_dir = './saved_models'
task_name = 'coatt'

torch.manual_seed(0)
log = Logger(20, task_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, dataset, criterion, optimizer):
    model.train()
    train_loss = 0
    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch['image'], batch['question'])
        loss = criterion(output, label)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    return train_loss / len(dataset_train)


def evaluate_split(model, dataset, criterion):
    model.eval()
    eval_loss = 0
    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    for batch in dataloader:
        with torch.no_grad():
            output = model(batch['image'], batch['question'])
            loss = criterion(output, label)
            eval_loss += loss.item()

    return eval_loss / len(dataset_train)


def train(model, dataset_train, dataset_val):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    log.info('Number of steps per epoch: {}'.format(len(dataset_train)))
    print('Number of steps per epoch: {}'.format(len(dataset_train)))

    pat = patience
    best_valid_loss = None
    for epoch in range(num_epochs):
        log.info('Epoch: {}'.format(epoch))
        print('Epoch: {}'.format(epoch))

        train_loss = train_epoch(model, dataset_train, criterion, optimizer)
        log.info('Training loss: {:.3f}'.format(train_loss))
        print('Training loss: {:.3f}'.format(train_loss))

        valid_loss = eval_samples(model, dataset_val, criterion)
        log.info('Validation loss: {:.3f}'.format(valid_loss))
        print('Validation loss: {:.3f}'.format(valid_loss))

        model.save(os.path.join(save_dir, '{}-{}.pt'.format(task_name, epoch)))
        if not best_valid_loss or valid_loss < best_valid_loss:
            model.save(os.path.join(save_dir, '{}-best.pt'.format(task_name)))
            best_valid_loss = valid_loss
            pat = patience
        else:
            pat -= 1

        if pat == 0: # Early stopping
            break

    log.info('Finished training!')
    print('Finished training!')


def evaluate(model, dataset_test):
    criterion = nn.NLLLoss()
    model.load(os.path.join(save_dir, '{}-best.pt'.format(task_name))).to(device)

    test_loss = eval_samples(model, dataset_test, criterion)
    log.info('Test loss: {:.3f}'.format(test_loss))
    print('Test loss: {:.3f}'.format(test_loss))


dataset_train = VQADataset(data_dir, 'train')
dataset_val = VQADataset(data_dir, 'val')
dataset_test = VQADataset(data_dir, 'test')

vocab = VQAVocab(data_dir)

model = CoNet(len(vocab.qvocab), emb_dim, len(vocab.avocab), useco=use_coatt)

for name, param in model.named_parameters():
    print(name, param.requires_grad, param.is_cuda, param.size())
    # assert param.is_cuda

train(model, dataset_train, dataset_val)
evaluate(model, dataset_test)
