import torch
import torch.nn as nn
import torch.nn.functional as F


class Encode(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(Encode, self).__init__()
        self.mlp = nn.Linear(embedding_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def view(self, sentence, batch_size):
        return self.mlp(sentence).view(batch_size, -1, self.hidden_dim)

    def forward(self, a, b):
        batch_size = a.shape[0]
        return self.view(a, batch_size), self.view(b, batch_size)


class SelfAttention(nn.Module):
    def __init__(self,embedding_dim, hidden_dim, output_dim, dropout_rate=0.0):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.F = self.mlp(hidden_dim, hidden_dim, dropout_rate)
        self.G = self.mlp(hidden_dim * 2, hidden_dim, dropout_rate)
        self.H = self.mlp(hidden_dim * 2, hidden_dim, dropout_rate)
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
        a_size = a.shape[1]
        b_size = b.shape[1]
        # Attend
        # Unnormalized attention weights a = F(a), b = F(b), e = F(a) * Ft(b)
        a = self.F(a.view(-1, self.hidden_dim))
        a = a.view(-1, a_size, self.hidden_dim)
        b = self.F(b.view(-1, self.hidden_dim))
        b = b.view(-1, b_size, self.hidden_dim)

        e = torch.bmm(a, torch.transpose(b, 1, 2))

        # normalize attention weights (alpha and beta from the paper)
        beta = torch.bmm(F.softmax(e, dim=2), b)
        alpha = torch.bmm(F.softmax(torch.transpose(e, 1, 2), dim=2), a)

        # Compare
        # concat a with beta, b with alpha and feed through G
        a_cat_beta = torch.cat((a, beta), dim=2)
        b_cat_alpha = torch.cat((b, alpha), dim=2)
        v1 = self.G(a_cat_beta.view(-1, 2 * self.hidden_dim))
        v2 = self.G(b_cat_alpha.view(-1, 2 * self.hidden_dim))

        # Aggregate
        # Sum over each value from the comparison (reducing dimension back to hidden_dim)
        # Feed through the final mlp
        v1 = torch.sum(v1.view(-1, a_size, self.hidden_dim), dim=1)
        v2 = torch.sum(v2.view(-1, b_size, self.hidden_dim), dim=1)

        v1 = torch.squeeze(v1, dim=1)
        v2 = torch.squeeze(v2, dim=1)

        v1_cat_v2 = torch.cat((v1, v2), dim=1)
        h = self.H(v1_cat_v2)
        
        # This two phases are not done in the paper, it worked for us.
        # flat h to the output's dimension (number of labels)
        out = self.output(h)

        # normalize results
        return self.softmax(out)
