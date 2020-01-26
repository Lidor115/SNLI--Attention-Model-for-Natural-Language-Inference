import torch
import torch.nn as nn
import torch.nn.functional as F


class Encode(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encode, self).__init__()
        #torch.manual_seed(3)
        #self.embed = nn.Embedding(vocab_size, embedding_dim)
        #nn.init.uniform_(self.embed.weight, -1.0, 1.0)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

    def view(self, sentence, batch_size):
        return self.mlp(sentence).view(batch_size, -1, self.hidden_dim)

    def forward(self, a, b):
        batch_size = a.shape[0]
        embedded_a = self.embed(a).view(-1, self.embedding_dim)
        embedded_b = self.embed(b).view(-1, self.embedding_dim)
        return self.view(embedded_a, batch_size), self.view(embedded_b, batch_size)


class SelfAttention(nn.Module):
    def __init__(self,embedding_dim, hidden_dim, output_dim, dropout_rate=0.0):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.F = self.mlp(hidden_dim, hidden_dim, dropout_rate)
        self.G = self.mlp(hidden_dim * 2, hidden_dim, dropout_rate)
        self.H = self.mlp(hidden_dim * 2, hidden_dim, dropout_rate)
        self.embed = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    @staticmethod
    def mlp(input_dim, output_dim, dropout_rate):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, a, b):
        batch_size = a.shape[0]
        a= self.embed(a).view(batch_size,-1,self.hidden_dim)
        b= self.embed(b).view(batch_size,-1,self.hidden_dim)
        l_a = a.shape[1]
        l_b = b.shape[1]

        a = self.F(a.view(-1, self.hidden_dim))  # a: ((batch_size * l_a) x hidden_dim)
        a = a.view(-1, l_a, self.hidden_dim)  # a: (batch_size x l_a x hidden_dim)
        b = self.F(b.view(-1, self.hidden_dim))  # b: ((batch_size * l_b) x hidden_dim)
        b = b.view(-1, l_b, self.hidden_dim)  # b: (batch_size x l_b x hidden_dim)

        # equation (1) in paper:
        e = torch.bmm(a, torch.transpose(b, 1, 2))  # e: (batch_size x l_a x l_b)

        # equation (2) in paper:
        beta = torch.bmm(F.softmax(e, dim=2), b)  # beta: (batch_size x l_a x hidden_dim)
        alpha = torch.bmm(F.softmax(torch.transpose(e, 1, 2), dim=2), a)  # alpha: (batch_size x l_b x hidden_dim)

        # equation (3) in paper:
        a_cat_beta = torch.cat((a, beta), dim=2)
        b_cat_alpha = torch.cat((b, alpha), dim=2)
        v1 = self.G(a_cat_beta.view(-1, 2 * self.hidden_dim))  # v1: ((batch_size * l_a) x hidden_dim)
        v2 = self.G(b_cat_alpha.view(-1, 2 * self.hidden_dim))  # v2: ((batch_size * l_b) x hidden_dim)

        # equation (4) in paper:
        v1 = torch.sum(v1.view(-1, l_a, self.hidden_dim), dim=1)  # v1: (batch_size x 1 x hidden_dim)
        v2 = torch.sum(v2.view(-1, l_b, self.hidden_dim), dim=1)  # v2: (batch_size x 1 x hidden_dim)

        v1 = torch.squeeze(v1, dim=1)
        v2 = torch.squeeze(v2, dim=1)

        # equation (5) in paper:
        v1_cat_v2 = torch.cat((v1, v2), dim=1)  # v1_cat_v2: (batch_size x (hidden_dim * 2))
        h = self.H(v1_cat_v2)

        out = self.output(h)

        return self.softmax(out)
