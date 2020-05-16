
import utils


# _test()


# T = 10
# B = 8
# H = 32
#
# pos = [1, 0, 4, 7, 3, 6, 5, 2]
# outputs = torch.rand((T, B, H)).to(torch.device('cuda'))
# outputs_b = torch.zeros_like(outputs)
# old = outputs.clone().detach().cpu()
#
# for index_t, _ in enumerate(old):
#     for i, index in enumerate(pos):
#         outputs_b[index_t][index] = old[index_t][i]
#
# outputs_b = outputs_b.cpu()
# outputs = outputs.cpu()
#
# print(outputs == outputs_b)
#
# new_outputs = torch.zeros_like(outputs)
# for i, index in enumerate(pos):
#     new_outputs[:][index] = outputs[:][i]
#
# print(new_outputs == outputs_b)


# import numpy as np
# import torch
# import itertools
#
# outputs = [
#     [1, 1, 1, 1],
#     [2, 2, 2],
#     [3, 3, 3, 3, 3],
#     [4, 4]
# ]
# outputs, seq_lens, pos = utils.sort_batch(outputs)
# print(outputs)
# print(seq_lens)
# print(pos)
#
# rev_pos = np.argsort(pos)
# print(rev_pos)
#
# outputs = list(itertools.zip_longest(*outputs, fillvalue=0))
# outputs = [list(b) for b in outputs]
# outputs = torch.tensor(outputs)
#
# old_outputs = outputs.clone()
#
# outputs = torch.index_select(outputs, 1, torch.tensor(rev_pos))
# print(outputs)
#
# old_outputs = old_outputs.transpose(0, 1)
# new_outputs = old_outputs.clone()
# for i, index in enumerate(pos):
#     new_outputs[index][:] = old_outputs[i][:]
# new_outputs = new_outputs.transpose(0, 1)
#
# print(new_outputs == outputs)


# import torch
#
# outputs = [[2, 3, 4, 43, 23, 3561, 234, 234, 234, 2]]
# outputs = torch.tensor(outputs)
# print(outputs.shape)
#
# topv, topi = outputs.topk(1)
# print(topv.shape)
# print(topi)


# import torch
#
# T = 10
# B = 8
# nl_vocab_size = 32
#
# nl_batch = torch.randint(0, nl_vocab_size, [T, B])
# decoder_outputs = torch.rand(T, B, nl_vocab_size)
#
# criterion_1 = torch.nn.NLLLoss()
# criterion_1.eval()
#
# criterion_2 = torch.nn.NLLLoss()
# criterion_2.eval()
#
# with torch.no_grad():
#     # sentence
#     loss = 0
#     for i in range(B):
#         nl = nl_batch[:][i]
#         decoder_output = decoder_outputs[:][i][:]
#         loss += criterion_1(decoder_output, nl)
#     loss = loss / T
#     print(loss.item())
#
#     # batch
#     nl_batch = nl_batch.view(-1)
#     decoder_outputs = decoder_outputs.view(-1, nl_vocab_size)
#     batch_loss = criterion_2(decoder_outputs, nl_batch)
#     print(batch_loss.item())

# import torch
# from torch.utils.data import Dataset, DataLoader
#
#
# class MyDataset(Dataset):
#     def __init__(self, size):
#         self.x = torch.randn(size, 1)
#
#     def __getitem__(self, index):
#         return self.x[index]
#
#     def __len__(self):
#         return len(self.x)
#
#
# dataset = MyDataset(1001)
#
# data_loader = DataLoader(dataset,
#                          batch_size=10)
#
# print(len(data_loader))
#
# data_loader = DataLoader(dataset,
#                          batch_size=10,
#                          drop_last=True)
#
# print(len(data_loader))

