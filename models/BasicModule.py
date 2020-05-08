# coding=utf-8
import torch as t
import time
import os

class BasicModule(t.nn.Module):
    
    def __init__(self):
        """
        封装了nn.Module，主要提供save和load两个方法
        """
        super(BasicModule, self).__init__()
        self.model_name = self.__class__.__name__
    
    def load(self, path):
        """
        可加载指定路径的模型
        """
        self.load_state_dict(t.load(path))
    
    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名，
        如AlexNet_0710_23:57:29.pth
        """
        if name is None:
            prefix = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints/' + self.model_name + '_')
            now_time = time.strftime('%m%d_%Hh%Mm%Ss.pth')
            name = prefix + now_time
        t.save(self.state_dict(), f=name)
        print(f"model have been saved at checkpoints/{self.model_name}_{now_time}\n")
        return name