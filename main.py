# coding=utf-8
import torch as t
from torch.utils.data import DataLoader
from torchnet import meter
import os
from gensim.models import KeyedVectors
import time

from utils import Visualizer
from data import Sentence, sentence2data
from config import opt
import models


def setup_seed(seed):
     t.manual_seed(seed)
     t.cuda.manual_seed_all(seed)
     t.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)


def train(**kwargs):
    print(kwargs)
    start = time.time()
    # 根据命令行参数更新配置
    vis = Visualizer(opt.env)
    opt.parse(kwargs)

    # 加载词向量
    print("Loading word vectors...Please wait.")
    vector = KeyedVectors.load_word2vec_format(
        os.path.join(os.path.dirname(os.path.realpath(opt.train_data_root)), 'vector.txt')
    )
    print("Successfully loaded word vectors.")

    # step1: 模型
    model = getattr(models, opt.model)(input_size=vector.vector_size+2, output_size=opt.class_num)
    if opt.load_model_path:
        model.load(opt.load_model_path) # 预加载
    if opt.use_gpu and t.cuda.is_available():
        model = model.cuda()
    print(f"Structure of {model.model_name}:\n{model}\n")

    # step2: 数据
    train_data = Sentence(root=opt.train_data_root, 
                          relations=opt.relations, 
                          max_length=opt.max_length,
                          vector=vector,
                          train=True)   # 训练集
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True) 

    val_data = Sentence(opt.train_data_root, opt.relations, opt.max_length, vector, train=False)  # 验证集
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=True)

    # step3: 目标函数和优化器
    loss_fn = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(params=model.parameters(), 
                             lr=lr,
                             weight_decay = opt.weight_decay)

    # step4: 统计指标：平滑处理之后的损失，还有混淆矩阵
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(opt.class_num)
    previous_loss = 1e100
    
    # 训练
    for epoch in range(opt.max_epoch):
        
        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label) in enumerate(train_dataloader):

            # 训练模型参数 
            input = data
            target = label
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            prediction = model(input)
            loss = loss_fn(prediction, target)
            loss.backward()
            optimizer.step()
            
            # 更新统计指标以及可视化
            loss_meter.add(loss.item())
            confusion_matrix.add(prediction.data, target.data)

            # if ii % opt.print_freq == opt.print_freq - 1:
            #     vis.plot('train loss', loss_meter.value()[0])
                
                # 如果需要的话，进入debug模式
                # if os.path.exists(opt.debug_file):
                #     import ipdb;
                #     ipdb.set_trace()
        cm_value = confusion_matrix.value()
        correct = 0
        for i in range(cm_value.shape[0]):
            correct += cm_value[i][i]
        accuracy = 100. * correct / (cm_value.sum())

        vis.plot('train loss', loss_meter.value()[0])
        vis.plot('train accuracy', accuracy)

        if epoch % opt.save_epoch == opt.save_epoch -1:
            model.save()

        # 计算验证集上的指标及可视化
        val_lm, val_cm, val_accuracy = val(model, val_dataloader)
        vis.plot('val loss', val_lm.value()[0])
        vis.plot('val accuracy', val_accuracy)

        
        print("epoch:{epoch}, lr:{lr}, loss:{loss}\ntrain_cm:\n{train_cm}\nval_cm:\n{val_cm}"
                .format(epoch=epoch,
                        loss=loss_meter.value()[0],
                        val_cm=str(val_cm.value()),
                        train_cm=str(confusion_matrix.value()),
                        lr=lr)
        )
        
        # 如果损失不再下降，则降低学习率
        if loss_meter.value()[0] > previous_loss:          
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
        previous_loss = loss_meter.value()[0]

    cost = int(time.time()) - int(start)
    print(f"Cost {int(cost/60)}min{cost%60}s.")


def val(model, dataloader):
    """
    计算模型在验证集上的准确率等信息
    """

    # 把模型设为验证模式
    model.eval()
    
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(opt.class_num)
    for ii, data in enumerate(dataloader):
        val_input, val_label = data
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)
        loss = t.nn.functional.cross_entropy(score, val_label)

        loss_meter.add(loss.item())
        confusion_matrix.add(score.data.squeeze(), val_label.long())

    # 把模型恢复为训练模式
    model.train()
    
    cm_value = confusion_matrix.value()
    correct = 0
    for i in range(cm_value.shape[0]):
        correct += cm_value[i][i]
    accuracy = 100. * correct / (cm_value.sum())
    return loss_meter, confusion_matrix, accuracy


def predict(**kwargs):
    opt.parse(kwargs)

    # 加载词向量
    print("Loading word vectors...Please wait.")
    vector = KeyedVectors.load_word2vec_format(
        os.path.join(os.path.dirname(os.path.realpath(opt.train_data_root)), 'vector.txt')
    )
    print("Successfully loaded word vectors.")

    # 模型
    model = getattr(models, opt.model)(input_size=vector.vector_size+2, output_size=opt.class_num)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu and t.cuda.is_available():
        model = model.cuda()
    
    # 循环接收请求
    while True:

        # 数据
        company_pair = input("Please input a pair of companies(separate them with a space):\n")
        
        if company_pair == '':
            print("See you next time!")
            break

        try:
            org1, org2 = company_pair.split()
        except ValueError:
            print("Input Error! Please try again.")
            continue
        while True:
            text = input("Input a sentence continuing these two companies:\n")
            if org1 not in text or org2 not in text:
                print("These two companies are not included in the sentence. Try again.")
                continue
            break
        pos = (text.index(org1), text.index(org2)) # 公司位置
        data = sentence2data(text, pos, vector)
        input_data = t.from_numpy(data).view(-1, data.shape[0], data.shape[1]) # 增加一个维度
        if opt.use_gpu and t.cuda.is_available():
            input_data = input_data.cuda()

        # 预测
        out = model(input_data)
        prediction = t.nn.functional.softmax(out)
        maxminum = prediction.max(dim=1) # 最大值
        relation = opt.relations[maxminum[1].item()]
        probability = maxminum[0].item()
        print(f"Prob: {probability}\t{relation}\n")



if __name__ == '__main__':
    import fire
    fire.Fire()
    # train()
