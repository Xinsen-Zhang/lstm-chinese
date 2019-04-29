from torch import nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self,vocab_size ,time_step,hidden_size= 32):
        super(LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=self.vocab_size, 
            hidden_size=hidden_size,
            batch_first= True,
            num_layers= 1
            )
        self.dropout= nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_size, self.vocab_size)
        # self.m = nn.Softmax()

    def forward(self, X, hidden):
        # embeded = self.embedding(X)
        # embeded =  embeded.view(embeded.shape[0], embeded.shape[1], -1)
        # print(embeded.shape)
        out1, hidden = self.lstm(X, hidden)
        out_ = out1.contiguous().view(-1, self.hidden_size)
        out2 = self.linear(out_)
        # out = self.m(out2)
        # out = F.log_softmax(out2, dim= 0)
        return out2, hidden

