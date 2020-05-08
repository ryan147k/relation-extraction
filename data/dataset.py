# coding=utf-8
import os
from gensim.models import KeyedVectors
import torch as t
from torch.utils.data import Dataset
import re
import numpy as np
import jieba


class Sentence(Dataset):
    
    def __init__(self, root, relations, max_length, vector, train:True):
        """
        max_length 表示句子的最大长度，多了去掉，少了填充（为了mini-batch）
        """
        super(Sentence, self).__init__()
        self.relations = relations
        self.max_length = max_length
        self.train = train

        if self.train:
            filename = 'train_data.txt'
        else:
            filename = 'val_data.txt'
        
        sentences = open(os.path.join(os.path.realpath(root), filename), 'r', encoding='utf-8').readlines()
        self.sents = sentences
        self.vector = vector
    
    def __getitem__(self, index):
        org1, org2, relation, text = self.sents[index].split('\t')

        label = self.relations.index(relation)
        pos = (text.index(org1), text.index(org2)) # org1,org2在text中首次出现的位置

        data = sentence2data(text, pos, self.vector)

        # 判断长度，切割或填充
        if len(data) > self.max_length:
            data = data[:self.max_length]
        else:
            data = np.append(data, np.zeros([self.max_length-len(data), data.shape[1]]), axis=0)
        return data, label

    def __len__(self):
        return len(self.sents)


def sentence2data(sentence, pos, vector):
    """
    句子转成输入矩阵
    pos: 实体位置
    vector: 词向量
    """
    data = []
    words = jieba.lcut(sentence)
    for i, c in enumerate(words):
        pos_embedding = [i-pos[0], i-pos[1]] # 位置嵌入就是当前字离两个实体的距离
        try:
            char_embedding = vector[c]
        except KeyError:
            # 如果c是未登录字，用它前一个字代替它
            char_embedding = data[i-1][:-2] if i > 0 else np.zeros(vector.vector_size, dtype=np.float64)
        char_embedding = np.append(char_embedding, pos_embedding)
        data.append(char_embedding)
    data = np.array(data)

    # 判断长度，切割或填充
    # if len(data) > self.max_length:
    #     data = data[:self.max_length]
    # else:
    #     data = np.append(data, np.zeros([self.max_length-len(data), data.shape[1]]), axis=0)
    return data
