import torch
from torch import nn
import math

class multi_head_attention(nn.Module):
    def __init__(self, d_model, n_head):
        super(multi_head_attention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        B, T, D = q.shape
        n_d = self.d_model // self.n_head
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # split
        q = q.view(B, T, self.n_head, n_d).transpose(1, 2)
        k = k.view(B, T, self.n_head, n_d).transpose(1, 2)
        v = v.view(B, T, self.n_head, n_d).transpose(1, 2)

        # scaled dot prodction
        score = q @ k.transpose(2, 3) / math.sqrt(n_d)
        mask = torch.tril(torch.ones(T, T, dtype=bool))  # 创建一个下三角掩码
        # 将掩码中为0的位置的得分设置为-10000。这是为了在应用softmax函数时，这些位置的得分接近于0
        score = score.masked_fill(mask == 0, -10000)
        score = self.softmax(score)
        score = score @ v  

        # concate
        x_concate = score.transpose(1, 2).contiguous().view(B, T, self.d_model)
        x_output = self.w_o(x_concate)
        return x_output

if __name__ == '__main__':
    X = torch.randn(16, 64, 512)  # 模拟输入，batch_size=16, 序列长度=64, 隐藏单元=512
    d_model = 512  # 隐藏单元
    n_head = 8  # 多头注意力机制的头数
    attn = multi_head_attention(d_model, n_head)
    Y = attn(X, X, X)
    print(Y.shape)
