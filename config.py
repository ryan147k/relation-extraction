# coding=utf-8
import warnings
import os


class DefaultConfig():
    # 数据集参数
    train_data_root = './data/train/'
    relations = [
        line.strip('\n') for line in open(os.path.join(os.path.realpath(train_data_root), 'relation.txt'),
                                          mode='r', 
                                          encoding='utf-8').readlines()
    ]
    class_num = len(relations)
    max_length = 80 # 数据中一个句子的最大长度

    # 模型参数
    env = 'default'
    model = 'BiLSTM'
    load_model_path = None # 加载预训练的模型的路径，为None代表不加载

    # 训练参数
    max_epoch = 50
    save_epoch = 5 # 多少个epoch保存一次模型
    batch_size = 128
    use_gpu = True
    lr = 1e-4
    lr_decay = 0.95 # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4 # 损失函数
    print_freq = 20 # print info every N batch


def parse(self, kwargs):
    """
    根据字典kwargs 更新 config参数
    """
    # 更新配置参数
    for k, v in kwargs.items():
        if not hasattr(self, k):
            # 警告还是报错，取决于你个人的喜好
            warnings.warn("Warning: opt has not attribut %s" %k)
        if k == 'train_data_root':  # 如果更改了训练路径，相应的relations,class_num也要更改
            relations = [
                line.strip('\n') for line in open(os.path.join(os.path.realpath(v), 'relation.txt'),
                                          mode='r', 
                                          encoding='utf-8').readlines()
            ]
            setattr(self, 'relations', relations)
            setattr(self, 'class_num', len(relations))
        setattr(self, k, v)
        
    # 打印配置信息	
    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()