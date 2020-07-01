import os
import json

import numpy as np
from skimage import io
from skimage.transform import resize
from skimage.color import gray2rgb
import torch
from torch.utils.data import Dataset

class VQADataset(Dataset):
    """Contains VQA dataset v2.0."""

    def __init__(self, data_dir, split):
        """
        Args:
            data_dir (string): Path to the dataset folder.
            split (string): train/val/test
        """
        assert split in ('train', 'val', 'test')
        exfn = os.path.join(data_dir, '{}.json'.format(split))
        with open(exfn, 'r') as f:
            self.examples = json.load(f)
        self.data_dir = data_dir
        self.split = split

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        example = self.examples[idx]
        qid = example['question_id'] # For evaluation
        question = example['question']
        answer = example['answer']
        image_name = os.path.join(os.path.join(os.path.join(self.data_dir, 'images'), self.split), example['image_name'])
        image = resize(io.imread(image_name), (512, 512))
        if len(image.shape) == 2:
            image = gray2rgb(image)

        return {'question_id': qid, 'image': image, 'question': question, 'answer': answer}


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = VQADataset('./data', 'val')
    print(len(dataset))

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4)
    for i, batch in enumerate(dataloader):
        print(i, batch['image'].size(), batch['question'], batch['answer'])
        if i == 3:
            break
