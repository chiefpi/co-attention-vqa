"""Build vocabulary for questions and answers."""
import os
import json

PAD_TOK = '<pad>' # PAD_ID = 0

class Vocab(object):
    def __init__(self):
        self.id2token = []
        self.token2id = {}
        self.add_token(PAD_TOK)

    def __len__(self):
        return len(self.id2token)

    def add_token(self, token):
        if token not in self.token2id:
            self.id2token.append(token)
            self.token2id[token] = len(self.id2token) - 1


class VQAVocab(object):
    def __init__(self, data_dir):
        self.qvocab = Vocab()
        self.avocab = Vocab()
        self.data_dir = data_dir
        for split in 'train', 'val', 'test':
            self.add_vocab(split)
        
    def add_vocab(self, split):
        exfn = os.path.join(self.data_dir, '{}.json'.format(split))
        with open(exfn, 'r') as f:
            examples = json.load(f)
        qtokens = []
        atokens = []
        for ex in examples:
            qtokens += ex['question'].split()
            atokens.append(ex['answer']) # No split
        for token in qtokens:
            self.qvocab.add_token(token)
        for token in atokens:
            self.avocab.add_token(token)


if __name__ == "__main__":
    vocab = VQAVocab('./data')
    print(len(vocab.qvocab), len(vocab.avocab))
    print(vocab.qvocab.id2token[:100])
    print(vocab.avocab.id2token[:100])