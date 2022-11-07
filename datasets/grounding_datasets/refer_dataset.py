# -*- coding: utf-8 -*-

"""
Copied from https://github.com/zyang-ur/ReSC/blob/e4022f87bfd11200b67c4509bb9746640834ceae/utils/transforms.py

ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.
Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

import sys
import cv2
import torch
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
from collections import OrderedDict
sys.path.append('.')

from transformers import BertTokenizerFast, RobertaTokenizerFast


cv2.setNumThreads(0)

def build_bert_tokenizer(bert_model):
    if bert_model.split('-')[0] == 'roberta':
        lang_backbone = RobertaTokenizerFast.from_pretrained(bert_model, do_lower_case=True, do_basic_tokenize=False)
    else:
        lang_backbone = BertTokenizerFast.from_pretrained(bert_model, do_lower_case=True, do_basic_tokenize=False)
    return lang_backbone

class DatasetNotFoundError(Exception):
    pass


class ReferSegDataset(Dataset):
    SUPPORTED_DATASETS = {
        'refcoco_unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'refcoco+_unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'refcocog_google': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'refcocog_umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
        }
    }

    def __init__(self, data_root, im_dir, seg_dir, dataset='refcoco_unc', 
                 split='train', max_query_len=40, bert_model='bert-base-uncased'):
        super(ReferSegDataset, self).__init__()
        self.images = []
        self.data_root = data_root
        self.im_dir = im_dir
        self.dataset = dataset
        self.query_len = max_query_len
        self.split = split
        self.tokenizer = build_bert_tokenizer(bert_model)

        dataset_dir = self.dataset.split('_')[0]
        annotation_path = osp.join(data_root, dataset_dir)
        self.seg_dir = osp.join(seg_dir, dataset_dir)

        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']
        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        splits = [split]
        splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(annotation_path, imgset_file)
            self.images += torch.load(imgset_path)

    def pull_item(self, idx):
        img_file, seg_file, bbox, phrase = self.images[idx]
        ## box format: x1y1x2y2
        bbox = np.array(bbox, dtype=int)
        img = cv2.imread(osp.join(self.im_dir, img_file))
        mask = np.load(osp.join(self.seg_dir, seg_file))
        assert img.shape[:2] == mask.shape[:2]
        ## duplicate channel if gray image
        if img.shape[-1] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.stack([img] * 3)
        return img, mask, phrase, bbox, img_file

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, mask, phrase, bbox, img_file = self.pull_item(idx)
        # phrase = phrase.decode("utf-8").encode().lower()
        phrase = phrase.lower()

        # encode phrase to bert input
        tokenized_sentence = self.tokenizer(
            phrase,
            padding='max_length',
            max_length=self.query_len,
            truncation=True,
            return_tensors='pt',
        )
        word_id = tokenized_sentence['input_ids'][0]
        word_mask = tokenized_sentence['attention_mask'][0]

        h, w, c = img.shape

        samples = {
            "img": img,
            "sentence": np.array(word_id, dtype=int),
            "sentence_mask": np.array(word_mask, dtype=int),
            "phrase": phrase,
            "img_file": img_file
        }

        mask = mask[None, :, :]
        image_id = int(img_file.split('.')[0].split('_')[-1])
        target = {
            "image_id": image_id,
            'dataset_id': idx,
            "boxes": np.array([bbox], dtype=np.float32),
            "labels": [0],
            "masks": mask,
            "orig_size": np.array([h, w], dtype=np.int)
        }
        return samples, target

