import torch
import numpy as np
import importlib

# Helpers when setting up training

def importNet(net):
    """

    :param net: 用.分隔的模型路径字符串
    :return: 找到这个.分隔的模型路径中的最后一个名字，即模型名字，然后运行它？（直接写不行吗。。。）
    """
    t = net.split('.')
    path, name = '.'.join(t[:-1]), t[-1]
    module = importlib.import_module(path)
    return eval('module.{}'.format(name))

def make_input(t, requires_grad=False, need_cuda = True):
    # 就是将输入转化为对应的torch变量。
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    if need_cuda:
        inp = inp.cuda()
    return inp

def make_output(x):
    # 将torch变量变为numpy返回
    if not (type(x) is list):
        return x.cpu().data.numpy()
    else:
        return [make_output(i) for i in x]
