from torch.utils.data import DataLoader

from dataset import VQADataset
from vocab import VQAVocab
from model.conet import CoNet
from logger import Logger


# Parameters
data_dir = './data'
use_coatt = True
emb_dim = 300
feat_dim = 1024


def train(model, dataset_train, dataset_val):
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    for i, batch in enumerate(dataloader):
        pass


def evaluate(model, dataset_test):
    pass


dataset_train = VQADataset(data_dir, 'train')
dataset_val = VQADataset(data_dir, 'val')
dataset_test = VQADataset(data_dir, 'test')

vocab = VQAVocab(data_dir)

model = CoNet(len(vocab.qvocab), emb_dim, feat_dim, len(vocab.avocab), use_coatt)

train(model, dataset_train, dataset_val)
evaluate(model, dataset_test)
