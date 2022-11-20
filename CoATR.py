import numpy as np
from numpy import dtype, random, sqrt
from numpy.core.fromnumeric import trace
import torch
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
# from torch.nn.modules.conv import Conv2d
from torch.utils.data import dataloader


class TRDnet(torch.nn.Module):
    def __init__(self, shape, ranks, nc,  device="cpu", **kwargs):
        super(TRDnet, self).__init__()

        self.EmeddingLayers = torch.nn.ModuleList([torch.nn.Embedding(shape_i, embedding_dim=(ranks*ranks), padding_idx=0)
                                                   for shape_i in shape])
        for i, shape_i in enumerate(shape):
            self.EmeddingLayers[i].weight.data = torch.randn((shape_i, (ranks*ranks)), dtype=torch.float).to(device)

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.bne = torch.nn.BatchNorm2d(nc)

        self.conv2d_1 = torch.nn.Conv2d(1, nc, kernel_size=(ranks//2,ranks//2), stride = ranks//2)
        self.conv2d_2 = torch.nn.Conv2d(nc, nc, kernel_size=(1, len(shape)))
        self.conv2d_3 = torch.nn.Conv2d(nc, nc, kernel_size=(2, 2))
        self.conv2d_4 = torch.nn.Conv2d(nc, nc, kernel_size=(1, len(shape)))
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(in_features=nc, out_features=(nc//2))
        self.linear2 = torch.nn.Linear(in_features=(nc//2), out_features=1)

        self.rel_num = kwargs["AR_D"]
        self.conv_loss = torch.nn.Conv2d(1, 1, kernel_size=(self.rel_num, 1), bias=False)


    def forward(self, idex, ranks, Batch_size):

        batch_size, shape_size = idex.shape

        conv_Z = torch.cat([self.EmeddingLayers[shape_i](idex[:, shape_i]).view(batch_size, ranks, ranks) for shape_i in range(shape_size)],
                           dim=2).unsqueeze(1)


        rst = self.conv2d_1(conv_Z)
        rst = self.bne(rst)
        rst = F.relu(rst)
        rst = self.hidden_dropout(rst)
        rst = self.conv2d_2(rst) 
        rst = F.relu(rst)   
        rst = self.hidden_dropout(rst)
        rst = self.conv2d_3(rst)  
        rst = F.relu(rst)   
        rst = self.conv2d_4(rst)  
        rst = F.relu(rst)   
        rst = self.flatten(rst)

        input = rst.view(Batch_size, -1, rst.shape[-1])  # [Batch, 144, nc]
        B, T, _ = input.shape

        rst = self.linear1(rst) 
        rst = F.relu(rst)
        rst = self.linear2(rst) 



        # input = self.ln1(input)

        # output, _ = self.lstm1(input)
        # output = self.ln2(output)

        # output, _ = self.lstm2(output)
        # output = self.ln3(output)

        # output = torch.reshape(output, (-1, rst.shape[-1]))
        # output = self.linear1(output)
        # output = self.relu1(output)
        # output = self.linear2(output)


        # 算loss
        loss_out = rst.view((B, 1, T, 1))
        loss_a = loss_out[:, :, self.rel_num:, :].view((B, T - self.rel_num)) #砍掉了前rel_num个数据

        # B 1 T-k+1 1
        loss_b = self.conv_loss(loss_out)
        loss_b = loss_b[:, :, :-1, :].view((B, T - self.rel_num))

        # mse_loss_fn = torch.nn.MSELoss()
        # loss_rel = mse_loss_fn(loss_a, loss_b)
        loss_rel = (loss_a - loss_b) * (loss_a - loss_b)
        # loss_rel = torch.sqrt(loss_rel.sum(-1)).mean()
        loss_rel = torch.sqrt(loss_rel.mean())

        # rst = self.flatten(rst)
        # rst = self.linear1(rst) 
        # rst = F.relu(rst) 

        # rst = self.linear2(rst) 

        return rst, loss_rel

