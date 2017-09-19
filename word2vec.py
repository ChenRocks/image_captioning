import argparse
import logging
import os
import codecs
import json
from os.path import join, exists, basename, dirname, abspath
from time import time
from itertools import chain

import gensim
import torch
from torch.nn import init

from tokens import *


_SAVE_PATH = join(dirname(abspath(__file__)), 'embedding')
_COCO_ANN_PATH = join(dirname(abspath(__file__)),
                       '../data/coco/annotations/captions_train2014.json')


class Sentences:
    def __init__(self):
        with open(_COCO_ANN_PATH, 'r') as f:
            self._anns = json.loads(f.read())['annotations']

    def __iter__(self):
        for ann in self._anns:
            words = ann['caption'].lower().strip().split(' ')
            if words[-1][-1] == '.':
                words = words[:-1] + [words[-1][:-1]]
            words = [w.strip() for w in words if w.strip()]
            yield words + ['<\s>']


def main(args):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    st = time()
    if not exists(_SAVE_PATH):
        os.makedirs(_SAVE_PATH)

    sentences = Sentences()
    model = gensim.models.Word2Vec(size=args.dim, min_count=2, workers=8, sg=1)
    model.build_vocab(sentences)
    print('vocab_built, {:.0f} seconds elapsed'.format(time()-st))
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

    model.save(join(_SAVE_PATH, 'word2vec.{}d.{}k.bin'.format(
        args.dim, int(len(model.wv.vocab)/1000))))
    model.wv.save_word2vec_format(join(_SAVE_PATH, 'word2vec.{}d.{}k.w2v'.format(
        args.dim, int(len(model.wv.vocab)/1000))))

    print('word2vec trained in {:.0f} seconds'.format(time()-st))


def load_embedding_from_bin(bin_file, id2word):
    attrs = basename(bin_file).split('.')  #{algo}.{dim}d.{vsize}k.bin
    vocab_size = len(id2word)
    emb_dim = int(attrs[-3][:-1])
    embedding = torch.Tensor(vocab_size, emb_dim)
    init.uniform(embedding, -0.1, 0.1)
    oovs = []
    if attrs[-4] == 'word2vec':
        model = gensim.models.Word2Vec.load(bin_file)
        w2v = model.wv
    elif attrs[-4] == 'fasttext':
        w2v = gensim.models.wrappers.FastText.load(bin_file)
    else:
        raise ValueError()

    for i in range(vocab_size):
        if i == EOS:
            embedding[i, :] = torch.from_numpy(w2v['<\s>'])
        elif i == PAD:
            embedding[i, :].normal_(0, 1)
        elif id2word[i] in w2v:
            embedding[i, :] = torch.from_numpy(w2v[id2word[i]])
        else:
            oovs.append(i)

    return embedding, oovs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train word2vec embedding on coco captions'
    )
    parser.add_argument('--dim', action='store', type=int, required=True)
    args = parser.parse_args()

    main(args)



