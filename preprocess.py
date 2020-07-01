import os
import json
import nltk

data_dir = './data'
idir = './data/images'
qdir = './data/questions'
adir = './data/annotations'

def preprocess(split):
    """Link image names, questions, and answers in a single json file."""
    qfn = os.path.join(qdir, '{}.json'.format(split))
    afn = os.path.join(adir, '{}.json'.format(split))
    output_fn = os.path.join(data_dir, '{}.json'.format(split))

    id2q = {}
    with open(qfn, 'r') as f:
        questions = json.load(f)['questions']
        for q in questions:
            id2q[q['question_id']] = q['question']

    items = []
    with open(afn, 'r') as f:
        annotations = json.load(f)['annotations']
        split_name = 'val' if split == 'test' else split
        for anno in annotations:
            item = {
                'image_id': anno['image_id'],
                'image_name': 'COCO_{}2014_{:012d}.jpg'.format(split_name, anno['image_id']),
                'question_id': anno['question_id'],
                'question': ' '.join(nltk.word_tokenize(id2q[anno['question_id']])),
                'answer': anno['multiple_choice_answer']
            }
            
            if os.path.exists(os.path.join(os.path.join(idir, split), item['image_name'])):
                items.append(item)
    
    with open(output_fn, 'w') as f:
        json.dump(items, f)

for split in 'train', 'val', 'test':
    preprocess(split)