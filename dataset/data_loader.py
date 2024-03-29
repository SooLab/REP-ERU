# -*- coding: utf-8 -*-

"""
YouRefIt referring image PyTorch dataset.
Define and group batches of images and queries.
Based on:
https://github.com/zyang-ur/ReSC/blob/master/dataset/data_loader.py
"""
from torchvision.transforms import Compose, ToTensor, Normalize
import os
import sys
import cv2
import json
import uuid
import tqdm
import math
import torch
import random
# import h5py
import numpy as np
import os.path as osp
import scipy.io as sio
import torch.utils.data as data
from collections import OrderedDict
sys.path.append('.')
import operator
import utils
from utils import Corpus
import clip
import argparse
import collections
import logging
import json
import re

np.set_printoptions(threshold=np.inf)
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
# from transformers import BertTokenizer,BertModel
from utils.transforms import letterbox, random_affine

sys.modules['utils'] = utils
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(True)

def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples

## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features

class DatasetNotFoundError(Exception):
    pass

class ReferDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'yourefit': {'splits': ('train', 'val', 'test')},
        'referit': {'splits': ('train', 'val', 'trainval', 'test')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'gref_umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
        },
        'flickr': {
            'splits': ('train', 'val', 'test')}
    }

    def __init__(self, data_root, split_root='data', dataset='referit', imsize=256,
                 transform=None, augment=False, device=None, return_idx=False, testmode=False,
                 split='train', max_query_len=128, lstm=False, bert_model='bert-base-uncased'):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.imsize = imsize
        self.query_len = max_query_len
        self.lstm = lstm
        self.transform = transform
        self.testmode = testmode
        self.split = split
        self.device = device
        self.t = input_transform = Compose([
        ToTensor()
    ])
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.augment=augment
        self.return_idx=return_idx
        self.num = 0
        if self.dataset == 'yourefit':
            self.dataset_root = osp.join(self.data_root, 'yourefit')
            self.im_dir = osp.join(self.dataset_root, 'images')
        elif self.dataset == 'referit':
            self.dataset_root = osp.join(self.data_root, 'referit')
            self.im_dir = osp.join(self.dataset_root, 'images')
            self.split_dir = osp.join(self.dataset_root, 'splits')
        elif  self.dataset == 'flickr':
            self.dataset_root = osp.join(self.data_root, 'Flickr30k')
            self.im_dir = osp.join(self.dataset_root, 'flickr30k_images')
        else:   ## refcoco, etc.
            self.dataset_root = osp.join(self.data_root, 'other')
            self.im_dir = osp.join(
                self.dataset_root, 'images', 'mscoco', 'images', 'train2014')
            self.split_dir = osp.join(self.dataset_root, 'splits')

        if not self.exists_dataset():
            print('Please download index cache to data folder')
            exit(0)

        dataset_path = osp.join(self.split_root, self.dataset)
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        if self.lstm:
            self.corpus = Corpus()
            corpus_path = osp.join(dataset_path, 'corpus.pth')
            self.corpus = torch.load(corpus_path)

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        splits = [split]
        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}full.pth'.format(self.dataset, split)
            imgset_path = osp.join(dataset_path, imgset_file)
            self.images += torch.load(imgset_path)

    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))


    def pull_item(self, idx):
        if self.dataset == 'flickr':
            img_file, bbox, phrase = self.images[idx]
        else:
            img_file, _, bbox, phrase, attri = self.images[idx]
        ## box format: to x1y1x2y2
        if not (self.dataset == 'referit' or self.dataset == 'flickr'):
            bbox = np.array(bbox, dtype=int)
            #bbox[2], bbox[3] = bbox[0]+bbox[2], bbox[1]+bbox[3]
        else:
            bbox = np.array(bbox, dtype=int)
        
        img_path = osp.join(self.im_dir, img_file)
        img = cv2.imread(img_path)

        htmapdir = self.im_dir.replace('images', 'pafours')
        htmapfile = img_file #.replace('.jpg', '_rendered.png')
        htmap_path = osp.join(htmapdir, htmapfile)
        htmap = cv2.imread(htmap_path)
        
        ht = np.asarray(htmap)

        # #ht = np.mean(ht, axis=2)
         

        # ht = cv2.resize(ht, (512, 512))

        ptdir = self.im_dir.replace('images', 'depimg')
        ptfile = img_file #.replace('.jpg', '_depth.png')
        pt_path = osp.join(ptdir, ptfile)
        pt = cv2.imread(pt_path)
        # print(pt.shape)
        # exit()
        # pt = cv2.resize(pt, (256,256))
        # pt = np.reshape(pt, (3, 256, 256))

        saldir = self.im_dir.replace('images', 'saliency')
        salfile = img_file.replace('.jpg', '.jpeg')
        sal_path = osp.join(saldir, salfile)
        sal = cv2.imread(sal_path)
        sal = cv2.resize(sal, (256,256))
        #sal = np.reshape(sal, (3, 256, 256))

        gestdir = 'ln_data/bodysegment'
        gestfile = img_file.replace('.jpg' , '_seg.png')
        gest_path = osp.join(gestdir,gestfile)
        gest = cv2.imread(gest_path)
        if gest.shape != img.shape:
            gest = cv2.resize(gest, img.shape[:2], interpolation=cv2.INTER_AREA)
        ## duplicate channel if gray image
        if img.shape[-1] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.stack([img] * 3)
        
        return img, pt, ht, phrase, bbox, gest, sal, img_file
 #       return img, phrase, bbox, pt, ht

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, pt, ht, phrase, bbox, gest, sal, img_file = self.pull_item(idx)
        # phrase = phrase.decode("utf-8").encode().lower()


        phrase = phrase.lower()
        if self.augment:
            augment_flip, augment_hsv, augment_affine = True,True,True
        
        ## seems a bug in torch transformation resize, so separate in advance
        h,w = img.shape[0], img.shape[1]
        if self.augment:
            ## random horizontal flip
            if augment_flip and random.random() > 0.5:
                img = cv2.flip(img, 1)
                pt = cv2.flip(pt, 1 )
                ht = cv2.flip(ht, 1 )
                gest = cv2.flip(gest, 1)
                sal = cv2.flip(sal, 1 )
                bbox[0], bbox[2] = w-bbox[2]-1, w-bbox[0]-1
                phrase = phrase.replace('right','*&^special^&*').replace('left','right').replace('*&^special^&*','left')
   
            ## random intensity, saturation change
            if augment_hsv:
                fraction = 0.5
                img_hsv = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
                S = img_hsv[:, :, 1].astype(np.float32)
                V = img_hsv[:, :, 2].astype(np.float32)
                a = (random.random() * 2 - 1) * fraction + 1
                if a > 1:
                    np.clip(S, a_min=0, a_max=255, out=S)
                a = (random.random() * 2 - 1) * fraction + 1
                V *= a
                if a > 1:
                    np.clip(V, a_min=0, a_max=255, out=V)

                img_hsv[:, :, 1] = S.astype(np.uint8)
                img_hsv[:, :, 2] = V.astype(np.uint8)
                img = cv2.cvtColor(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB)
            
            mask = np.ones_like(img)
            img, mask, ratio, dw, dh = letterbox(img, mask, self.imsize)
            #ht, _, ratio, dw, dh = letterbox(ht, None, self.imsize)
            gest, _, ratio, dw, dh = letterbox(gest, None, self.imsize)
            #sal, _, ratio, dw, dh = letterbox(sal, None, self.imsize)
            bbox[0], bbox[2] = bbox[0]*ratio+dw, bbox[2]*ratio+dw
            bbox[1], bbox[3] = bbox[1]*ratio+dh, bbox[3]*ratio+dh
            ## random affine transformation
            if augment_affine:
                gt = np.asarray(torch.zeros([512,512]))
                gt[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 1
                img, mask, bbox, M = random_affine(img, mask, bbox, \
                    degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))
                pt = cv2.warpPerspective(pt, M, dsize=(512, 512), flags=cv2.INTER_LINEAR,
                              borderValue=0)
                ht = cv2.warpPerspective(ht, M, dsize=(512, 512), flags=cv2.INTER_LINEAR,
                              borderValue=0)
                gest = cv2.warpPerspective(gest, M, dsize=(512, 512), flags=cv2.INTER_NEAREST,
                              borderValue=0)
                sal = cv2.warpPerspective(sal, M, dsize=(256, 256), flags=cv2.INTER_NEAREST,
                              borderValue=0)
                gt = cv2.warpPerspective(gt, M, dsize=(512, 512), flags=cv2.INTER_NEAREST,
                              borderValue=0)
        else:   ## should be inference, or specified training
            mask = np.ones_like(img)
            img, mask, ratio, dw, dh = letterbox(img, mask, self.imsize)
            # ht, _, ratio, dw, dh = letterbox(ht, None, self.imsize)
            gest, _, ratio, dw, dh = letterbox(gest, None, self.imsize)
            bbox[0], bbox[2] = bbox[0]*ratio+dw, bbox[2]*ratio+dw
            bbox[1], bbox[3] = bbox[1]*ratio+dh, bbox[3]*ratio+dh
            gt = np.asarray(torch.zeros([512,512]))
            gt[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 1
        ## Norm, to tensor
        # print(img.shape)

        pt = pt[:,:,0]
        gest = gest[:,:,0]
        mask = mask[:,:,0]
        sal = np.reshape(sal, (3, 256, 256))
        sal = sal[0,:,:]
        if self.transform is not None:

            img = self.transform(img)
    
            #pt = self.t(pt)
            #print(ht.shape)

            ht = self.transform(ht)
 
            #print(ht.shape)
        if self.lstm:
            phrase = self.tokenize_phrase(phrase)
            word_id = phrase
            # word_mask = np.zeros(word_id.shape)
            word_mask = np.array(word_id>0,dtype=int)
        else:
            ## encode phrase to bert input
            
            examples = read_examples(phrase, idx)
            features = convert_examples_to_features(
                examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
            word_id = features[0].input_ids
            word_mask = features[0].input_mask
            #phrase = features[0].input_mask #clip.tokenize(phrase, context_length=20)
        if self.testmode:
            return img, pt, ht, gest, gt, mask, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
                np.array(bbox, dtype=np.float32), np.array(ratio, dtype=np.float32), \
                np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0],sal , phrase
        else:
            return img, pt, ht, gest, gt, mask, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
            np.array(bbox, dtype=np.float32),sal, phrase, img_file