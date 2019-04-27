from torch import nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size= 32,embedding_dim= 10):
        self.vocab_size = vocab_size
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_size,
            batch_first= True,
            num_layers= 1
            )
        self.dropout= nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, self.vocab_size)
        # self.m = nn.Softmax()

    def forward(self, X, hidden):
        embeded = self.embedding(X)
        embeded =  embeded.view(embeded.shape[0], embeded.shape[1], -1)
        # print(embeded.shape)
        out1, hidden = self.lstm(embeded, hidden)
        out2 = self.linear(out1)
        # out = self.m(out2)
        # out = F.log_softmax(out2, dim= 0)
        return out2, hidden

