import math
import random

import torch

from p3graph import Sample


class Dataset:

    def __init__(self, samples, batch_size):
        self.samples = samples
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.samples) / batch_size)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)
        batch = self.samples[index * self.batch_size: (index + 1) * self.batch_size]

        return batch

    def padding(self, samples:Sample):
        batch_size = len(samples)
        each_len_tensor = torch.tensor([len(s.label) for s in samples]).long()
        mx = torch.max(each_len_tensor).item()
        audio1_tensor = torch.zeros((batch_size, mx, 1024))
        audio2_tensor = torch.zeros((batch_size, mx, 1024))
        audio3_tensor = torch.zeros((batch_size, mx, 1024))
        visual1_tensor = torch.zeros((batch_size, mx, 2048))
        visual2_tensor = torch.zeros((batch_size, mx, 2048))
        visual3_tensor = torch.zeros((batch_size, mx, 2048))
        personality_tensor = torch.zeros((batch_size, mx, 768))

        labels = []
        bigfive = []
        for i, s in enumerate(samples):
            cur_len = len(s.label)
            audio1_tensor[i, :cur_len, :] = torch.stack([torch.from_numpy(t).float() for t in s.audio1])
            audio2_tensor[i, :cur_len, :] = torch.stack([torch.from_numpy(t).float() for t in s.audio2])
            audio3_tensor[i, :cur_len, :] = torch.stack([torch.from_numpy(t).float() for t in s.audio3])
            visual1_tensor[i, :cur_len, :] = torch.stack([torch.from_numpy(t).float() for t in s.visual1])
            visual2_tensor[i, :cur_len, :] = torch.stack([torch.from_numpy(t).float() for t in s.visual2])
            visual3_tensor[i, :cur_len, :] = torch.stack([torch.from_numpy(t).float() for t in s.visual3])
            personality_tensor[i, :cur_len, :] = torch.stack([torch.from_numpy(t).float() for t in s.personality])
            labels.extend(s.label)
            bigfive.extend(s.bigfive)

        label_tensor = torch.tensor(labels).long()
        bigfive_tensor = torch.tensor(bigfive).long()
        data = {

            "each_len_tensor": each_len_tensor,     # 每个batch不同样本的不同有效长度
            "audio1_tensor": audio1_tensor,
            "audio2_tensor": audio2_tensor,
            "audio3_tensor": audio3_tensor,
            "visual1_tensor": visual1_tensor,
            "visual2_tensor": visual2_tensor,
            "visual3_tensor": visual3_tensor,
            "personality_tensor": personality_tensor,
            "label_tensor": label_tensor,
            "bigfive_tensor": bigfive_tensor
        }
        return data

    def shuffle(self):
        random.shuffle(self.samples)




