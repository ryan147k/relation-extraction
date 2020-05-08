# coding=utf-8
from .BasicModule import BasicModule
import torch as t

class BiLSTM(BasicModule):
    
    def __init__(self, input_size, output_size):
        super(BiLSTM, self).__init__()

        self.bilstm = t.nn.LSTM(input_size=input_size,
                                hidden_size=64,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True)
        
        # 因为是双向LSTM,所以每个time_step输出是128维度
        self.out = t.nn.Linear(128, output_size)
    
    def forward(self, x):
        x = x.float()
        lstm_out, (h_n, h_c) = self.bilstm(x, None)
        out = self.out(lstm_out[:, -1, :]) # （batch, time_step, input）选每个样本的最后一个time_step的值
        return out