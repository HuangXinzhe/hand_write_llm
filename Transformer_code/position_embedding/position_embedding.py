"""
位置编码
"""
import torch
from matplotlib import pyplot as plt

dim = 512  # 位置编码的维度
pos_max = 1024  # 位置编码的最大长度
pos = torch.arange(pos_max).unsqueeze(1)  # 生成位置编码的位置信息
pos_embedding = torch.zeros(pos_max, dim)  # 初始化位置编码
base = 10000  # 位置编码的基数
pos_embedding[:, 0::2] = torch.sin(pos / (base ** (torch.arange(0, dim, 2) / dim)))
pos_embedding[:, 1::2] = torch.cos(pos / (base ** (torch.arange(0, dim, 2) / dim)))  

if __name__ == '__main__':
    # 可视化，查看位置编码的变化，以及位置编码的相似度，可以看到位置编码的相似度随着位置的增加而减小
    result = pos_embedding[0, :] @ pos_embedding.transpose(1, 0)
    result = result.tolist()
    x = range(pos_max)
    plt.plot(x, result)
    plt.show()
