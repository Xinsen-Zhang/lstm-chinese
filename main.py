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

epoch_num = 1
use_cuda = True if torch.cuda.is_available() else False
device = 'gpu' if use_cuda else 'cpu'

# ============= 实例化网络 ==========================
lstm = model.LSTM(VOCAB_NUM, hidden_size= hidden_size, embedding_dim= embedding_dim)

# ============= loss 和 optimizer 的定义 =============
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.005)

# ============ 设备的转移 ============================
if use_cuda:
    lstm.cuda()
    optimizer.cuda()

h_0 = torch.randn((1, batch_size,hidden_size))
c_0 = torch.randn((1, batch_size,hidden_size))
hidden = (h_0, c_0)
for  eooch in range(epoch_num):
    for i,(trainX, trainY) in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        trainX = Variable(torch.LongTensor(torch.from_numpy(trainX)))
        trainY = Variable(torch.LongTensor(torch.from_numpy(trainY)))
        # ========== 是否使用 gpu ====================
        if use_cuda:
            trainX.cuda()
            trainY.cuda()
        prediction_,hidden = lstm(trainX, hidden)
        trainY =  trainY.view(-1)
        prediction = prediction_.view(trainY.shape[0],-1)
        loss = criterion(prediction, trainY)
        loss.backward(retain_graph= True)
        optimizer.step()
        print('batch [{}/{}] loss {:.3f}'.format(i, batch_num, loss))
        if  i % 10 == 0:
            predictions = prediction.cpu().detach().numpy()
            predictions = np.argmax(predictions, axis= 1)
            predictions = ''.join([id2char[char] for char in predictions])
            print(predictions)
    break
            