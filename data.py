import random
import pickle as pkl
from collections import defaultdict
from os.path import dirname, join

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms

import numpy as np

from tokens import *


_FILE_PATH = dirname(__file__)
_COCO_IMG_PATH = join(_FILE_PATH, '../data/coco/{}2014')
_COCO_ANN_PATH = join(_FILE_PATH,
                       '../data/coco/annotations/captions_{}2014.json')
_COCO_WC_PATH = join(_FILE_PATH, '../data/coco/wc.pkl')

#_IMG_SIZE = 256
#_IMG_IN_SIZE = 224
_IMG_SIZE = 512
_IMG_IN_SIZE = 448
_IMG_MEAN = [0.485, 0.456, 0.406]
_IMG_STD = [0.229, 0.224, 0.225]


class CocoCaptionDataset(dset.CocoCaptions):
    def __init__(self, split, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split = split

    def __len__(self):
        l = super().__len__()
        if self.split == 'train':
            return l
        elif self.split == 'val':
            return int(l/3)
        else:
            return l - int(l/3)

    def __getitem__(self, i):
        if self.split in ['train', 'val']:
            ret = super().__getitem__(i)
        else:
            ret = super().__getitem__(int(super().__len__()/3)+i)
        return ret


def make_coco_cap_dataset(split):
    coco_split = split if split == 'train' else 'val'
    coco_cap = CocoCaptionDataset(
        split,
        root=_COCO_IMG_PATH.format(coco_split),
        annFile=_COCO_ANN_PATH.format(coco_split),
        transform=transforms.Compose([
            transforms.Scale(_IMG_SIZE), transforms.CenterCrop(_IMG_IN_SIZE),
            transforms.ToTensor(), transforms.Normalize(
                mean=_IMG_MEAN, std=_IMG_STD)
        ])
    )
    return coco_cap


def make_vocab(vocab_size):
    wc = pkl.load(open(_COCO_WC_PATH, 'rb'))
    word2id = {PAD_TOK: PAD, UNK_TOK: UNK, GO_TOK: GO, EOS_TOK: EOS}
    id2word = {PAD: PAD_TOK, UNK: UNK_TOK, GO: GO_TOK, EOS: EOS_TOK}
    for i, (w, _) in enumerate(wc.most_common(vocab_size), 4):
        word2id[w] = i
        id2word[i] = w
    return word2id, id2word


def pad_sequence(sequences):
    """ pad list of sequences to numpy array"""
    batch_size = len(sequences)
    max_len = len(max(sequences, key=lambda s: len(s)))
    batch = np.zeros((batch_size, max_len), dtype=np.int64)
    for i, s in enumerate(sequences):
        batch[i, :len(s)] = s
    return batch


def batch_caption(id_caps):
    """ pad target sentences batch for seq2seq"""
    inputs = [[GO] + cap for cap in id_caps]
    targets = [cap + [EOS] for cap in id_caps]
    input_ = torch.LongTensor(pad_sequence(inputs))
    target = torch.LongTensor(pad_sequence(targets))
    return input_, target


def get_imcap_collate_fn(word2id):
    def f(batches):
        imgs = [img for img, _ in batches]
        id_caps = [[[word2id[w] for w in cap.strip().split(' ')]
                    for cap in caps]
                   for _, caps in batches]
        ret =  list(zip(imgs, id_caps))
        assert(len(ret) > 0)
        assert(ret is not None)
        return ret
    return f


def coco_bucket_gen(loader, batch_size, bucket_size, word2id, max_cap_len):
    sort_fn = lambda pair: len(pair[0])
    for hyper_batch in loader:
        hyper_batch = [(im, c) for im, caps in hyper_batch for c in caps]
        hyper_batch = sorted(hyper_batch, key=sort_fn)
        indices = list(range(0, len(hyper_batch), batch_size))
        if loader.dataset.split=='train':
            random.shuffle(indices)
        for i in indices:
            batch = hyper_batch[i:i+batch_size]
            imgs = [b[0]for b in batch]
            captions = [b[1] for b in batch]
            yield imgs, captions


__TRAIN_DATASET = make_coco_cap_dataset('train')
def get_coco_train_loader(word2id, max_cap_len, batch_size, bucket_size=4, cuda=True):
    loader = DataLoader(
        __TRAIN_DATASET, batch_size=batch_size*bucket_size,
        shuffle=True, num_workers=2,
        collate_fn=get_imcap_collate_fn(word2id)
    )
    while True:
        batches = coco_bucket_gen(
            loader, batch_size, bucket_size, word2id, max_cap_len
        )
        for imgs, captions in batches:
            img = Variable(torch.stack(imgs, dim=0), volatile=True)
            input_, target = batch_caption(captions)
            input_ = Variable(input_)
            target = Variable(target)
            if cuda:
                img = img.cuda()
                input_ = input_.cuda()
                target = target.cuda()
            yield img, input_, target


__VAL_DATASET = make_coco_cap_dataset('val')
def get_coco_val_loader(word2id, max_cap_len, batch_size, bucket_size=4, cuda=True):
    loader = DataLoader(
        __VAL_DATASET, batch_size=batch_size*bucket_size,
        shuffle=False, num_workers=2,
        collate_fn=get_imcap_collate_fn(word2id)
    )
    def it():
        batches = coco_bucket_gen(
            loader, batch_size, bucket_size, word2id, max_cap_len
        )
        for imgs, captions in batches:
            img = Variable(torch.stack(imgs, dim=0), volatile=True)
            input_, target = batch_caption(captions)
            input_ = Variable(input_, volatile=True)
            target = Variable(target, volatile=True)
            if cuda:
                img = img.cuda()
                input_ = input_.cuda()
                target = target.cuda()
            yield img, input_, target
    return it


def coll(batches):
    imgs = [img for img, _ in batches]
    return imgs


def get_coco_test_iter(word2id, max_cap_len, batch_size, split='test', cuda=True):
    dataset = make_coco_cap_dataset(split)
    loader = DataLoader(
        dataset, batch_size=1,
        shuffle=False, num_workers=1,
        collate_fn=coll
    )
    ids = iter(dataset.ids[:len(dataset)] if split=='val'
               else dataset.ids[-len(dataset):])
    for id_, imgs in zip(ids, loader):
        img = Variable(torch.stack(imgs, dim=0), volatile=True)
        if cuda:
            img = img.cuda()
        yield id_, img


