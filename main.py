from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import argparse
import pickle as pkl
import math
import sys
import json
import os
import re
from time import time
from os.path import dirname, join, exists
from collections import OrderedDict, defaultdict

import torch
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

from tensorboard_logger import configure, log_value

from model.model import AttnImCap
from model.im import ResNetFeatExtracotr as ResNet
from tokens import *
from data import make_vocab
from data import get_coco_train_loader, get_coco_val_loader, get_coco_test_iter
from word2vec import load_embedding_from_bin


__SAVE_PATH = join(dirname(__file__), 'workspace')


def sequence_loss(logits, targets, size_average=True, pad_idx=0):
    """ functional interface of SequenceLoss"""
    batch_size, max_len, depth = logits.size()
    b, l = targets.size()
    assert(batch_size == b)
    assert(max_len == l)

    mask = targets != pad_idx
    target = targets.masked_select(mask)
    logit = logits.masked_select(
        mask.unsqueeze(2).expand_as(logits)
    ).contiguous().view(-1, depth)
    loss = F.cross_entropy(logit, target, size_average=size_average)

    assert(not math.isnan(loss.data[0]) and not math.isinf(loss.data[0]))
    return loss


def train_step(model, img, input_, target, optimizer, max_grad):
    model.train()
    optimizer.zero_grad()
    output = model(img, input_)
    loss = sequence_loss(output, target)
    loss.backward()
    params = (p for m in model.children() if not isinstance(m, ResNet)
              for p in m.parameters() if p.requires_grad)
    grad_norm = clip_grad_norm(params, max_grad)
    if grad_norm >= 100:
        print('WARNING: Exploding Gradients {:.2f}'.format(grad_norm))
        grad_norm = 100
    optimizer.step()
    return loss.data[0], grad_norm


def save_ckpt(model, loss, step, path):
    save_path = join(__SAVE_PATH, '{}/ckpt'.format(path))
    file_name = 'ckpt-{:4f}-{}'.format(loss, step)
    torch.save(model.state_dict(), join(save_path, file_name))


def get_best_ckpt(path):
    save_path = join(__SAVE_PATH, '{}/ckpt'.format(path))
    ckpts = os.listdir(save_path)
    ckpt_matcher = re.compile('^ckpt-.*-[0-9]*')
    best_ckpt = sorted([c for c in ckpts if ckpt_matcher.match(c)],
                       key=lambda c: float(c.split('-')[1]))[0]
    return join(save_path, best_ckpt)


def validate(model, loader):
    st = time()
    model.eval()
    val_loss = 0
    n = 0
    #for i in range(n_step):
        #img, input_, target = next(loader)
    for img, input_, target in loader():
        output = model(img, input_)
        # sum up batch loss
        val_loss += sequence_loss(
            output, target, size_average=False).data[0]
        n += (target != PAD).long().sum().data[0]

    val_loss /= n
    print('finished in {:.0f} seconds, Average loss: {:.4f}\n'.format(
        time()-st, val_loss))
    return val_loss


def test(model, loader, id2word, max_len):
    st = time()
    model.eval()
    results = []
    vizs = {}
    for i, (id_, img) in enumerate(loader):
        print('{}\r'.format(i), end='')
        output, attns = model.decode(img, GO, EOS, max_len)
        d = {}
        d['image_id'] = id_
        d['caption'] = ' '.join([id2word[i] for i in output])
        vizs[id_] = torch.stack(attns).cpu().numpy()
        #print('====================================')
        #print(id_)
        #print(d['caption'])
        results.append(d)

    print('finished in {:.0f} seconds\n'.format(time()-st))
    return results, vizs


def decode(args):
    with open(join(__SAVE_PATH, join(args.dir, 'vocab.pkl')), 'rb') as f:
        word2id, id2word = pkl.load(f)
    model = AttnImCap(len(id2word), args.emb_dim, args.n_cell, args.n_layer)
    if args.cuda:
        model.cuda()

    test_loader = get_coco_test_iter(
        args.max_len, args.batch_size,
        split=args.split, cuda=args.cuda)
    model.load_state_dict(torch.load(get_best_ckpt(args.dir)))
    results, vizs = test(model, test_loader, id2word, args.max_len)
    save_path = join(join(__SAVE_PATH, args.dir), args.split)
    if not exists(save_path):
        os.makedirs(save_path)
    with open(join(save_path, 'result.json'), 'w') as f:
        json.dump(results, f)
    with open(join(save_path, 'viz.pkl'), 'wb') as f:
        pkl.dump(vizs, f, pkl.HIGHEST_PROTOCOL)


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage
import skimage.transform as transform
from PIL import Image
import numpy as np
def plot_viz(img, cap, attn, path):
    words = cap.split(' ')
    n_words = len(words)
    w = round(math.sqrt(n_words))
    h = math.ceil(n_words / w)
    fig = plt.figure()
    plt.imshow(img)
    plt.axis('off')
    fig.savefig(join(path, 'img.png'))
    plt.subplot(w, h, 1)
    for i, a in enumerate(attn):
        plt.subplot(w, h, i+1)
        word = words[i]
        plt.text(0, 1, word, backgroundcolor='white', fontsize=13)
        plt.text(0, 1, word, color='black', fontsize=13)
        plt.imshow(img)
        alpha_img = transform.pyramid_expand(a,  upscale=32, sigma=20)
        plt.imshow(alpha_img, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    fig.savefig(join(path, 'viz.png'))


from random import choice
from torchvision.transforms import Scale, CenterCrop
def visualize(path):
    save_path = join(join(__SAVE_PATH, path), 'val')
    with open(join(save_path, 'result.json'), 'r') as f:
        results = json.load(f)
    with open(join(save_path, 'viz.pkl'), 'rb') as f:
        vizs = pkl.load(f)
    result = choice(results)
    result = results[0]
    id_ = result['image_id']
    cap = result['caption']
    attn = vizs[id_]
    img = Image.open('../data/coco/val2014/COCO_val2014_{0:012}.jpg'.format(id_))
    img = CenterCrop(448)(Scale(512)(img))
    save_path = join(__SAVE_PATH, path)
    plot_viz(img, cap, attn, save_path)


def main(args):
    if not exists(join(__SAVE_PATH, args.dir)):
        os.makedirs(join(__SAVE_PATH, args.dir))
    os.makedirs(join(__SAVE_PATH, '{}/ckpt'.format(args.dir)))
    word2id, id2word = make_vocab(args.vsize)
    with open(join(__SAVE_PATH, join(args.dir, 'vocab.pkl')), 'wb') as f:
        pkl.dump((word2id, id2word), f, pkl.HIGHEST_PROTOCOL)
    word2id = defaultdict(lambda: UNK, word2id)
    train_loader = get_coco_train_loader(
        word2id, args.max_len, args.batch_size, cuda=args.cuda)
    val_loader = get_coco_val_loader(
        word2id, args.max_len, args.batch_size, cuda=args.cuda)

    model = AttnImCap(len(id2word), args.emb_dim, args.n_cell, args.n_layer)
    if args.emb:
        emb, oovs = load_embedding_from_bin(args.emb, id2word)
        model.set_embedding(emb, oovs=oovs)
    if args.cuda:
        model.cuda()

    if args.opt == 'adam':
        opt_cls = optim.Adam
    else:
        raise ValueError()
    opt_kwargs = {'lr': args.lr}  # TODO
    optimizer = opt_cls(model.parameters(), **opt_kwargs)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=0, factor=0.5, verbose=True)

    meta = vars(args)
    with open(join(__SAVE_PATH, '{}/meta.json'.format(args.dir)), 'w') as f:
        json.dump(meta, f)
    configure(join(__SAVE_PATH, args.dir))
    step = 0
    running = None
    best_val = None
    patience = 0
    for img, input_, target in train_loader:
        loss, grad_norm = train_step(model, img, input_, target, optimizer, args.clip_grad)
        step += 1
        running = 0.99 * running + 0.01 * loss if running else loss
        log_value('loss', loss, step)
        log_value('grad', grad_norm, step)
        print('step: {}, running loss: {:.4f}\r'.format(step, running), end='')
        sys.stdout.flush()
        if step % args.ckpt_freq == 0:
            print('\nstart validation...')
            val_loss = validate(model, val_loader)
            log_value('val_loss', val_loss, step)
            save_ckpt(model, val_loss, step, args.dir)
            scheduler.step(val_loss)
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                patience = 0
            else:
                print('val loss does not decrease')
                patience += 1
            if patience > args.patience:
                break

    print('training finished, run test set')
    test_loader = get_coco_test_iter(
        args.max_len, args.batch_size, cuda=args.cuda)
    model.load_state_dict(torch.load(get_best_ckpt(args.dir)))
    result = test(model, test_loader, id2word, args.max_len)
    with open(join(__SAVE_PATH, '{}/result.json'.format(args.dir)), 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--decode', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--ckpt-size', type=int)
    parser.add_argument('--dir', required=True, metavar='DIR',
                        help='dir name to save')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--vsize', type=int, default=10000, metavar='V',
                        help='vocabulary size (default: 10000)')
    parser.add_argument('--emb_dim', type=int, default=256, metavar='E',
                        help='dimension of the word embedding (default: 256)')
    parser.add_argument('--emb', metavar='PE',
                        help='specify pretrained embedding')
    parser.add_argument('--n_cell', type=int, default=512, metavar='NC',
                        help='number of LSTM units (default: 512)')
    parser.add_argument('--n_layer', type=int, default=1, metavar='NL',
                        help='number of LSTM layers (default: 1)')
    parser.add_argument('--max_len', type=int, default=20, metavar='L',
                        help='maximum caption length (default: 20)')
    parser.add_argument('--opt', default='adam', metavar='OPT',
                        help='optimizer (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--clip_grad', type=float, default=2.0, metavar='G',
                        help='gradient clipping (default: 2.0)')
    parser.add_argument('--ckpt_freq', type=int, default=3000, metavar='F',
                        help='# of iteration per checkpoint')
    parser.add_argument('--patience', type=int, default=3, metavar='P',
                        help='# of checkpoint to wait for early stopping')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.split = 'test' if args.test else 'val'
    torch.backends.cudnn.benchmark = True
    # image will be cropped to same size, run benchmark to speed up conv
    if args.decode:
        import torch.multiprocessing as mp
        mp.set_start_method('forkserver')
        decode(args)
    elif args.viz:
        visualize(args.dir)
    else:
        import torch.multiprocessing as mp
        mp.set_start_method('forkserver')
        main(args)


