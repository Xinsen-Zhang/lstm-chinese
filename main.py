import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch.autograd import Variable
from LSTM import model
from utils import VOCAB_NUM,dataloader, id2char, batch_num
from config import BATCH_SIZE as batch_size
from config import HIDDEN_SIZE as hidden_size
from config import EMBEDDING_DIM as embedding_dim
from config import LR as lr
from config import TIME_STEPS as time_steps
import os
import codecs

epoch_num = 1
use_cuda = True if torch.cuda.is_available() else False
device = 'gpu' if use_cuda else 'cpu'

# ============= 实例化网络 ==========================
try:
    os.mkdir('checkpoint')
except Exception as e:
    pass
try:
    lstm = torch.load('./checkpoint/lstm.pkl')
    h_0 = torch.load('./checkpoint/h_n.pkl')
    c_0 = torch.load('./checkpoint/c_n.pkl')
    hidden = (h_0, c_0)
except Exception as e:
    lstm = model.LSTM(VOCAB_NUM, hidden_size= hidden_size, embedding_dim= embedding_dim)
    h_0 = Variable(torch.randn((1, batch_size,hidden_size)), requires_grad= True)
    c_0 = Variable(torch.randn((1, batch_size,hidden_size)), requires_grad= True)
    hidden = (h_0, c_0)

# ============= loss 和 optimizer 的定义 =============
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.005)

# ============ 设备的转移 ============================
if use_cuda:
    lstm.cuda()

# h_0 = Variable(torch.randn((1, batch_size,hidden_size)), requires_grad= True)
# c_0 = Variable(torch.randn((1, batch_size,hidden_size)), requires_grad= True)
# hidden = (h_0, c_0)
# hidden = None
for i,(trainX, trainY) in tqdm(enumerate(dataloader)):
    optimizer.zero_grad()
    trainX_ = Variable((torch.from_numpy(trainX)))
    trainY_ = Variable((torch.from_numpy(trainY)))
    # ========== 是否使用 gpu ====================
    if use_cuda:
        trainX_ = trainX_.cuda()
        trainY_ = trainY_.cuda()
        if hidden:
            hidden = (hidden[0].cuda(), hidden[1].cuda())
    prediction,hidden = lstm(trainX_, hidden)
    loss = criterion(prediction.view((batch_size * time_steps, -1)), trainY_.view((batch_size * time_steps)).long())
    loss.backward(retain_graph=True)
    nn.utils.clip_grad_norm_(lstm.parameters(), 5)
    optimizer.step()
    if i % 50 == 0:
        print('batch [{}/{}] loss: {:.3f}'.format(i, batch_num, loss))
        predictions = prediction.cpu().detach().numpy()
        predictions = np.argmax(predictions, axis= 1)
        predictions = ''.join([id2char[char] for char in predictions.flatten()])
        print(predictions)
print('batch [{}/{}] loss {:.3f}'.format(i, batch_num, loss))
torch.save(lstm, './checkpoint/lstm.pkl')
torch.save(hidden[0], './checkpoint/h_n.pkl')
torch.save(hidden[1], './checkpoint/c_n.pkl')
predictions = prediction.cpu().detach().numpy()
predictions = np.argmax(predictions, axis= 1)
predictions = ''.join([id2char[char] for char in predictions.flatten()])
print(predictions)
conntent_ = '\nbatch [{}/{}] loss {:.3f}'.format(i, batch_num, loss) + '\n{}'.format(predictions)
try:
    f = codecs.open('./checkpoint/log.txt', 'r', encoding='utf8')
    epoch = int(f.readline().split(':')[1])
    content = 'epoch:{}'.format(epoch + 1)
    f.close()
    f = codecs.open('./checkpoint/log.txt', 'w', encoding='utf8')
    f.write(content + conntent_)
    f.close()
except Exception as e:
    f = codecs.open('./checkpoint/log.txt', 'w', encoding='utf8')
    content = 'epoch:{}'.format(1)
    f.write(content + conntent_)
    f.close()