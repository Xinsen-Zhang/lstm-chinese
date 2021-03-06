
# -*- coding: utf-8 -*-
import os
import sys
num_epoch = int(sys.argv[1])
try:
    size_batch = int(sys.argv[2])
except Exception as e:
    pass

import codecs
import re
# =============== 读取文本内容 ===============
f = codecs.open('./data/xiaoaojianghu.txt', 'r', encoding='utf-8')
data = f.readlines()
# data = ''.join(data)

#=============== 简单的预处理 ===============
# 替换括号里的内容
pattern = re.compile(r'\(.*?\)')
data = [pattern.sub('', line) for line in data]

# 删除章节名称
pattern = re.compile(r'.*?第.*?[卷/章].*')
def isNotChapterName(text):
    if pattern.search(text):
        return False
    else:
        return True
data = [word for word in data if isNotChapterName(word)]


data = ''.join(data)


# 删除本站
pattern = re.compile(r'.*?本站.*')
def isNotChapterName(text):
    if pattern.search(text):
        return False
    else:
        return True
data = [word for word in data if isNotChapterName(word)]


# 省略号 => .
data = [line.replace('……', '。') for line in data]


# ==============判断char是否是乱码===================
def is_uchar(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'一' and uchar<=u'龥':
            return True
    """判断一个unicode是否是数字"""
    if uchar >= u'0' and uchar<=u'9':
            return True       
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'A' and uchar<=u'Z') or (uchar >= u'a' and uchar<=u'z'):
            return True
    if uchar in ('，','。','：','？','“','”','！','；','、','《','》','——'):
            return True
    return False

# 将每行的list合成一个长字符串
data = ''.join(data)
data = [char for char in data if is_uchar(char)]
data = ''.join(data)

print(data[-20:])

"""# vocabulary bank generation"""

VOCAB_BANK = set(data)
ID2CHAR = dict(enumerate(VOCAB_BANK))
CHAR2ID = {c:i for i,c in enumerate(VOCAB_BANK)}
VOCAB_NUM = len(VOCAB_BANK)
DATA = [CHAR2ID[char] for char in data]

print(VOCAB_NUM)

print(''.join([ID2CHAR[id_] for id_ in DATA[:10]]))

"""# data generator"""

import numpy as np
def data_generator(data=DATA, batch_size= 32, time_step=20):
    data = np.array(data)
    data_ = np.roll(data, -1)
    char_num = len(data)
    char_num_per_batch = batch_size * time_step
    batch_num = char_num // char_num_per_batch
    for n in range(0, batch_num * char_num_per_batch, char_num_per_batch):
        arr = data[n: n + char_num_per_batch]
        arr_ = data_[n: n + char_num_per_batch]
        yield (arr.reshape(batch_size, time_step), arr_.reshape(batch_size, time_step))

# test the generator
dataloader = data_generator()
for (x,y) in dataloader:
    print(''.join([ID2CHAR[id_] for id_ in x.flatten()]))
    print(''.join([ID2CHAR[id_] for id_ in y.flatten()]))
    break



"""# 定义 CharRNN"""

import time
import torch
from torch import nn
from torch.autograd import Variable

class CharRNN(nn.Module):
    def __init__(self,
                 tokens = VOCAB_BANK,
                embedding_size=200,
                hidden_size=128,
                num_layers= 2,
                keep_prob= 0.5,
                batch_first= True,
                bidirectional = False):
        super(CharRNN, self).__init__()
        
        self.tokens = tokens
        self.vocab_num = len(self.tokens)
        
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        self.batch_first= batch_first
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(self.vocab_num,
                                     self.embedding_size)
        self.lstm = nn.LSTM(
            input_size= self.embedding_size,
            hidden_size= self.hidden_size,
            batch_first= self.batch_first,
            num_layers = self.num_layers,
            dropout= self.keep_prob,
            bidirectional = self.bidirectional
        )
        self.dropout = nn.Dropout(self.keep_prob)
        # if self.bidirectional:
        #     self.linear = nn.Linear(self.hidden_size * 2,self.vocab_num)
        # else:
        #     self.linear = nn.Linear(self.hidden_size,self.vocab_num)
        self.linear = nn.Linear(self.hidden_size,self.vocab_num)
        
    def forward(self,x, hidden):
        out1 = self.embedding(x)
        out2, hidden = self.lstm(out1, hidden)
        out3 = self.dropout(out2)
        # if self.bidirectional:
        #     out4 = out3.contiguous().view(-1, 2 * self.hidden_size)
        # else:
        #     out4 = out3.contiguous().view(-1, self.hidden_size)
        out4 = out3.contiguous().view(-1, self.hidden_size)
        out5 = self.linear(out4)
        return out5, hidden

net = CharRNN(embedding_size= 256,
             hidden_size= 512,
              num_layers= 2,
             keep_prob= 0.1)
print(net)
hidden = None

del net
del hidden
# del optimizer
torch.cuda.empty_cache()
try:
    if size_batch:
        batch_size = size_batch
except Exception as e:
    batch_size = 20
time_step = 200
embedding_size = 300
hidden_size = 512
epoch_num = num_epoch
bidirectional = False

from tqdm import tqdm

try:
    net = CharRNN(embedding_size= embedding_size,
                 hidden_size= hidden_size,
                  num_layers= 2,
                 keep_prob= 0.1,
                 bidirectional = bidirectional)
    net.load_state_dict(torch.load('./checkpoint/net.pkl'))
except Exception as e:
    net = CharRNN(embedding_size= embedding_size,
                 hidden_size= hidden_size,
                  num_layers= 2,
                 keep_prob= 0.1,
                 bidirectional = bidirectional)
    print(net)
use_gpu = True if torch.cuda.is_available() else False
if use_gpu:
    net.cuda()
net.train()
batch_num = len(DATA) // (batch_size * time_step)
criterion = nn.CrossEntropyLoss()
try:
    optimizer = torch.optim.Adagrad(net.parameters(), lr= 5e-2)
    optimizer.load_state_dict(torch.load('./checkpoint/optimizer.pkl'))
except Exception as e:
    optimizer = torch.optim.Adagrad(net.parameters(), lr= 5e-2)
try:
    hidden_0 = torch.load('./checkpoint/hidden_0.pkl')
    hidden_1 = torch.load('./checkpoint/hidden_0.pkl')
    if use_gpu:
        hidden_0 = hidden_0.cuda()
        hidden_1 = hidden_1.cuda()
    hidden = (hidden_0, hidden_1)
except Exception as e:
    hidden = None

for epoch in tqdm(range(epoch_num)):
    dataloader = data_generator(batch_size= batch_size,
                           time_step= time_step)
    start = time.time()
    for i,(x,y) in (enumerate(dataloader)):
        optimizer.zero_grad()
        input_tensor = Variable(torch.from_numpy(x)).long()
        labels = Variable(torch.from_numpy(y)).long().contiguous().view(batch_size * time_step)
        if use_gpu:
            input_tensor = input_tensor.cuda()
            labels = labels.cuda()
        prediction, hidden = net(input_tensor, hidden)
        hidden = (hidden[0].data, hidden[1].data)
        if use_gpu:
            hidden = (hidden[0].cuda(), hidden[1].cuda())
        loss = criterion(prediction, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        optimizer.step()
        if (i+0) % 25 == 0:
            end = time.time()
            precision = torch.sum(torch.argmax(prediction, dim=1) == labels).detach()
            precision = float(precision.data) / labels.shape[0]
            duration = end - start
#             print('epoch [{}/{}] batch [{}/{}] costs {:.2f} seconds, loss {:.3f} precision {:.2f}%'.format(epoch,
#                                                                                                            epoch_num,
#                                                                                                            i, 
#                                                                                                            batch_num, 
#                                                                                                            duration, 
#                                                                                                            loss.data, 
#                                                                                                            precision * 100
#                                                                                                           ))
            prediction_ids = torch.argmax(prediction, dim=1).cpu().detach().numpy()
            prediction_text = ''.join([ID2CHAR[id_] for id_ in prediction_ids])
#             print(prediction_text[:20])
            start = time.time()
    
    precision = torch.sum(torch.argmax(prediction, dim=1) == labels).detach()
    precision = float(precision.data) / labels.shape[0]
    duration = end - start
    print('epoch [{}/{}] done, loss {:.3f} precision {:.2f}%'.format(epoch,
                                                                   epoch_num,
                                                                   loss.data, 
                                                                   precision * 100
                                                                  ))
    prediction_ids = torch.argmax(prediction, dim=1).cpu().detach().numpy()
    prediction_text = ''.join([ID2CHAR[id_] for id_ in prediction_ids])
#     print(prediction_text[:20])

start_char = '悟'
from random import choice
def tell_story(net= net, start_setence= '悟', length= 100, hidden= None, top_k = None):
    story = []
    net.eval()
    for char in start_setence:
        story.append(char)
    for i in range(length):
        start_tensor = Variable(torch.from_numpy(np.array([CHAR2ID[char] for char in start_setence])))
        if use_gpu:
            start_tensor = start_tensor.cuda()
        prediction,hidden = net(start_tensor.contiguous().view(-1,1),hidden)
        hidden = (hidden[0].data, hidden[1].data)
        if use_gpu:
            hidden = (hidden[0].cuda(), hidden[1].cuda())
        p = torch.softmax(prediction, dim= 1).cpu()
        p = p[-1,:]
        if top_k == None:
            top_k = VOCAB_NUM
        choosen = choice(p.topk(top_k)[1].numpy().squeeze())
        start_char = ID2CHAR[choosen]
        story.append(start_char)
        start_setence = start_setence[1:] + start_char
    return ''.join(story)

content = tell_story(start_setence=data[-batch_size:],top_k = 2, hidden = hidden, length=100000)

# save_model
def save_model(content='demo'):
    if os.path.exists('./checkpoint'):
        pass
    else:
        os.mkdir('./checkpoint')
    torch.save(net.state_dict(), './checkpoint/net.pkl')
    torch.save(optimizer.state_dict(), './checkpoint/optimizer.pkl')
    torch.save(hidden[0], './checkpoint/hidden_0.pkl')
    torch.save(hidden[1], './checkpoint/hidden_1.pkl')
    if os.path.exists('./checkpoint/story.txt'):
        f = codecs.open('./checkpoint/story.txt', 'r', encoding='utf8')
        epoch_ = int(f.readline().split(':')[1])
        epoch_ += epoch_num
    else:
        epoch_ = epoch_num
    f = codecs.open('./checkpoint/story.txt', 'w', encoding='utf8')
    f.write('epoch:{}'.format(epoch_))
    f.close()
    f = codecs.open('./checkpoint/story_after_{}_epoch.txt'.format(epoch_), 'w', encoding='utf8')
    f.write(content)
    f.close()
    os.system('git add .')
    os.system('git commit -m "经过{}轮迭代之后生成的文本"'.format(epoch_))
save_model(content= content)

