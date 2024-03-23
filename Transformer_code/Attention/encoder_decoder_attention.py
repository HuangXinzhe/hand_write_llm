"""
encoder-decoder attention
只计算q和k之间的注意力分数
"""
import torch

# 将padding值设置为0
k_pad_idx = 0
q_pad_idx = 0

# token序列
# src: 第一句话长度为3，第二句话长度为4，
# 在这个batch中，batch_length=4，第一句话需要padding一个0
src_token = torch.tensor([[3, 4, 192, 0], [2, 8, 5, 3]])

# trg: 第一句话长度为2，第二句话长度为3
# 在这个batch中，batch_length=3，第一句话需要padding一个0
trg_token = torch.tensor([[6, 7, 0], [11, 28, 9]])

len_q, len_k = trg_token.size(1), src_token.size(1)


# embeding
# src = torch.randn(2, 4, 512) # 2个batch， 4个长度， 512维度
# trg = torch.randn(2, 3, 512) # 2个batch， 3个长度， 512维度

# 多头Q
# src encode k : 2个batch， 8个头， 4个长度， 单头64维度
src = torch.randn(2, 8, 4, 64)

# trg decode Q : 2个batch， 8个头， 3个长度， 单头64维度
trg = torch.randn(2, 8, 3, 64)


# 为配合多头注意力，需要填充维度
src_mask = src_token.ne(k_pad_idx).unsqueeze(1).unsqueeze(2)
trg_mask = trg_token.ne(q_pad_idx).unsqueeze(1).unsqueeze(3)

# 批量处理
src_mask_repeat = src_mask.repeat(1, 1, len_q, 1)
trg_mask_repeat = trg_mask.repeat(1, 1, 1, len_k)

mask = src_mask_repeat & trg_mask_repeat


# 计算 encode K， decode Q之间的注意力分数
q = trg
k_t = src.transpose(2, 3)

score = q @ k_t  # 此处score是q和k的点积

score_mask = score.masked_fill(mask == 0, -torch.inf)
# score_mask = score.masked_fill(mask == 0, -10000)
print(score_mask[0, 0, :,:])