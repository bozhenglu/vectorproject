import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F

batch_size=2

max_num_src_words=8
max_num_tgt_words=8
model_dim=8

#序列最大長度
max_num_src_len=5
max_num_tgt_len=5


src_len=torch.Tensor([2,4]).to(torch.int32)
tgt_len=torch.Tensor([4,3]).to(torch.int32)

src_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1, max_num_src_words, (L,)), (0, max_num_src_len-L), value=0), dim=0) for L in src_len])
tgt_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1, max_num_tgt_words, (L,)), (0, max_num_tgt_len-L), value=0), dim=0) for L in tgt_len])



src_embedding_table=nn.Embedding(max_num_src_words+1,model_dim) #model_dim:每個單詞向量的數量 eg:model_dim=3,[0.2,0.1,0.7]
tgt_embedding_table=nn.Embedding(max_num_tgt_words+1,model_dim) 
src_embedding=src_embedding_table(src_seq)




print(src_embedding_table.weight)
print(src_seq)
print(src_embedding)