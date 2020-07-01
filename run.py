import os
import json

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from dataset import VQADataset
from vocab import VQAVocab
from model.conet import CoNet
from logger import Logger


# Parameters
data_dir = './data'
task_name = 'coatt'
use_coatt = True
emb_dim = 300
feat_dim = 2048
num_epochs = 30
patience = 5
batch_size = 64
lr = 0.001
num_workers = 4
save_dir = './saved_models'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

torch.manual_seed(0)
log = Logger(20, task_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def token2id(seqs, qvocab):
    tensors = [torch.LongTensor([qvocab.token2id[t] for t in seq.split()]) for seq in seqs]
    return pad_sequence(tensors, batch_first=True)


def answer2id(answers, avocab):
    return torch.LongTensor([avocab.token2id[a] for a in answers])


def id2answer(ids, avocab):
    return [avocab.id2token[i] for i in ids]


def train_epoch(model, dataset, vocab, criterion, optimizer):
    model.train()
    train_loss = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        ibatch = batch['image'].permute(0, 3, 1, 2).float().to(device)
        qbatch = token2id(batch['question'], vocab.qvocab).to(device)
        output = model(ibatch, qbatch)

        abatch = answer2id(batch['answer'], vocab.avocab).to(device)
        loss = criterion(output, abatch)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    return train_loss / len(dataset)


def evaluate_split(model, dataset, vocab, criterion):
    model.eval()
    eval_loss = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    for batch in tqdm(dataloader):
        with torch.no_grad():
            ibatch = batch['image'].permute(0, 3, 1, 2).float().to(device)
            qbatch = token2id(batch['question'], vocab.qvocab).to(device)
            output = model(ibatch, qbatch)

            abatch = answer2id(batch['answer'], vocab.avocab).to(device)
            loss = criterion(output, abatch)
            eval_loss += loss.item()

    return eval_loss / len(dataset)


def train(model, dataset_train, dataset_val, vocab):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    log.info('Number of steps per epoch: {}'.format(len(dataset_train)))
    print('Number of steps per epoch: {}'.format(len(dataset_train)))

    pat = patience
    best_valid_loss = None
    for epoch in range(num_epochs):
        log.info('Epoch: {}'.format(epoch))
        print('Epoch: {}'.format(epoch))

        train_loss = train_epoch(model, dataset_train, vocab, criterion, optimizer)
        log.info('Training loss: {:.3f}'.format(train_loss))
        print('Training loss: {:.3f}'.format(train_loss))

        valid_loss = evaluate_split(model, dataset_val, vocab, criterion)
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


def evaluate(model, dataset, vocab):
    results = []
    model.load(os.path.join(save_dir, '{}-best.pt'.format(task_name))).to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    for batch in dataloader:
        idbatch = batch['question_id']
        ibatch = batch['image'].permute(0, 3, 1, 2).to(device)
        qbatch = token2id(batch['question'], vocab.qvocab).to(device)
        output = model(ibatch, qbatch)
        aids = output.argmax(-1).tolist()
        answers = id2answer(aids, vocab.avocab)

        for i in range(batch_size):
            results.append({
                'question_id': idbatch[i],
                'answer': answers[i]
            })

    with open('results.json', 'w') as f:
        json.dump(results, f)
    print ('Finished evaluation!')


if __name__ == "__main__":
    dataset_train = VQADataset(data_dir, 'train')
    dataset_val = VQADataset(data_dir, 'val')
    dataset_test = VQADataset(data_dir, 'test')

    vocab = VQAVocab(data_dir)

    model = CoNet(len(vocab.qvocab), emb_dim, len(vocab.avocab), useco=use_coatt).to(device)

    for name, param in model.named_parameters():
        print(name, param.requires_grad, param.is_cuda, param.size())
        # assert param.is_cuda

    train(model, dataset_train, dataset_val, vocab)
    evaluate(model, dataset_test, vocab)
