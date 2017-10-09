import random
import json
import pickle as pkl
from collections import defaultdict
from os.path import dirname, join

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

import numpy as np

from tokens import *


_FILE_PATH = dirname(__file__)
_COCO_IMG_PATH = join(_FILE_PATH, '../data/coco/{}2014')
_COCO_ANN_PATH = join(_FILE_PATH,
                      '../data/coco/annotations/captions_{}2014.json')
_TEST_META_PATH = join(_FILE_PATH,
                      '../data/coco/annotations/image_info_test2014.json')
_COCO_WC_PATH = join(_FILE_PATH, '../data/coco/wc.pkl')

#_IMG_SIZE = 256
#_IMG_IN_SIZE = 224
_IMG_SIZE = 512
_IMG_IN_SIZE = 448
_IMG_MEAN = [0.485, 0.456, 0.406]
_IMG_STD = [0.229, 0.224, 0.225]

_VAL_SPLIT = 1

class CocoCaptionDataset(dset.CocoCaptions):
    def __init__(self, split, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split = split

    def __len__(self):
        l = super().__len__()
        if self.split == 'train':
            return l
        elif self.split == 'val':
            return int(l/_VAL_SPLIT)
        else:
            return l - int(l/_VAL_SPLIT)

    def __getitem__(self, i):
        if self.split in ['train', 'val']:
            ret = super().__getitem__(i)
        else:
            ret = super().__getitem__(int(super().__len__()/_VAL_SPLIT)+i)
        return ret

class CocoTestDataset(Dataset):
    def __init__(self, transforms=None):
        super().__init__()
        self._meta = json.loads(open(_TEST_META_PATH).read())['images']
        self._trans = transforms

    def __len__(self):
        return len(self._meta)

    def __getitem__(self, i):
        id_ = self._meta[i]['id']
        img = Image.open(
            join(_COCO_IMG_PATH.format('test'), self._meta[i]['file_name'])
        ).convert('RGB')
        if self._trans:
            img = self._trans(img)
        return id_, img




def make_coco_cap_dataset(split):
    coco_split = split if split == 'train' else 'val'
    trans = transforms.Compose([
        transforms.Scale(_IMG_SIZE), transforms.CenterCrop(_IMG_IN_SIZE),
        transforms.ToTensor(), transforms.Normalize(
            mean=_IMG_MEAN, std=_IMG_STD)
    ])
    coco_cap = CocoCaptionDataset(
        coco_split,
        root=_COCO_IMG_PATH.format(coco_split),
        annFile=_COCO_ANN_PATH.format(coco_split),
        transform=trans
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

def _split_caps(caps):
    splited_caps = []
    for cap in caps:
        words = cap.lower().strip().split(' ')
        if words[-1][-1] == '.':
            words = words[:-1] + [words[-1][:-1]]
        splited_caps.append([w.strip() for w in words if w.strip()])
    return splited_caps

def imcap_collate_fn(batches): # fixes mp problem
    imgs = [img.cuda() for img, _ in batches]
    cap_words = [_split_caps(caps) for _, caps in batches]
    ret =  list(zip(imgs, cap_words))
    return ret


def get_imcap_collate_fn(word2id):
    def split_caps(caps):
        splited_caps = []
        for cap in caps:
            words = cap.lower().strip().split(' ')
            if words[-1][-1] == '.':
                words = words[:-1] + [words[-1][:-1]]
            splited_caps.append([w.strip() for w in words if w.strip()])
        return splited_caps
    def f(batches):
        imgs = [img for img, _ in batches]
        cap_words = [split_caps(caps) for _, caps in batches]
        id_caps = [[[word2id[w] for w in cap] for cap in caps]
                   for caps in cap_words]
        ret =  list(zip(imgs, id_caps))
        return ret
    return f


def coco_bucket_gen(loader, batch_size, bucket_size, max_cap_len):
    sort_fn = lambda pair: len(pair[1])
    for debug_i, hyper_batch in enumerate(loader):
        hyper_batch = [(im, c) for im, caps in hyper_batch for c in caps]
        hyper_batch = sorted(hyper_batch, key=sort_fn)
        indices = list(range(0, len(hyper_batch), batch_size))
        random.shuffle(indices)
        for i in indices:
            batch = hyper_batch[i:i+batch_size]
            imgs = [b[0]for b in batch]
            captions = [b[1] for b in batch]
            yield imgs, captions


def get_coco_train_loader(word2id, max_cap_len, batch_size, bucket_size=4, cuda=True):
    dataset = make_coco_cap_dataset('train')
    loader = DataLoader(
        dataset, batch_size=batch_size*bucket_size,
        shuffle=True, num_workers=2 if cuda else 0,
        collate_fn=imcap_collate_fn if cuda else get_imcap_collate_fn(word2id)
    )
    while True:
        batches = coco_bucket_gen(
            loader, batch_size, bucket_size, max_cap_len
        )
        for imgs, captions in batches:
            img = Variable(torch.stack(imgs, dim=0), volatile=True)
            if cuda:
                captions = [[word2id[w] for w in cap] for cap in captions]
            input_, target = batch_caption(captions)
            input_ = Variable(input_)
            target = Variable(target)
            if cuda:
                input_ = input_.cuda()
                target = target.cuda()
            yield img, input_, target


def get_coco_val_loader(word2id, max_cap_len, batch_size, bucket_size=4, cuda=True):
    dataset = make_coco_cap_dataset('val')
    loader = DataLoader(
        dataset, batch_size=batch_size*bucket_size,
        shuffle=False, num_workers=2 if cuda else 0,
        collate_fn=imcap_collate_fn if cuda else get_imcap_collate_fn(word2id)
    )
    def f():
        batches = coco_bucket_gen(
            loader, batch_size, bucket_size, max_cap_len
        )
        for imgs, captions in batches:
            img = Variable(torch.stack(imgs, dim=0), volatile=True)
            if cuda:
                captions = [[word2id[w] for w in cap] for cap in captions]
            input_, target = batch_caption(captions)
            input_ = Variable(input_, volatile=True)
            target = Variable(target, volatile=True)
            if cuda:
                input_ = input_.cuda()
                target = target.cuda()
            yield img, input_, target
    return f


def coll(batches):
    imgs = [img for img, _ in batches]
    return imgs

def test_coll(batches):
    return batches[0][0], batches[0][1].unsqueeze(0).cuda()


def get_coco_test_iter(max_cap_len, batch_size, split='test', cuda=True):
    if split == 'val':
        dataset = make_coco_cap_dataset(split)
        loader = DataLoader(
            dataset, batch_size=1,
            shuffle=False, num_workers=1,
            collate_fn=coll
        )
        #ids = iter(dataset.ids[:len(dataset)] if split=='val'
                   #else dataset.ids[-len(dataset):])
        for id_, imgs in zip(dataset.ids, loader):
            img = Variable(torch.stack(imgs, dim=0), volatile=True)
            if cuda:
                img = img.cuda()
            yield id_, img
    else:
        trans = transforms.Compose([
            transforms.Scale(_IMG_SIZE), transforms.CenterCrop(_IMG_IN_SIZE),
            transforms.ToTensor(), transforms.Normalize(
                mean=_IMG_MEAN, std=_IMG_STD)
        ])
        dataset = CocoTestDataset(trans)
        loader = DataLoader(
            dataset, batch_size=1,
            shuffle=False, num_workers=1,
            collate_fn=test_coll
        )
        for id_, img in loader:
            img = Variable(img, volatile=True)
            if cuda:
                img = img.cuda()
            yield id_, img



