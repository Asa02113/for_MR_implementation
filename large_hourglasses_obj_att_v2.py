# ------------------------------------------------------------------------------
# This code is base on 
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn

class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

class fully_connected(nn.Module):
    def __init__(self, inp_dim, out_dim, with_bn=True):
        super(fully_connected, self).__init__()
        self.with_bn = with_bn

        self.linear = nn.Linear(inp_dim, out_dim)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x):
        linear = self.linear(x)
        bn     = self.bn(linear) if self.with_bn else linear
        relu   = self.relu(bn)
        return relu

class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)
        
        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)


class hourglasses(nn.Module):
    def __init__(self, nstack):
        super(hourglasses, self).__init__()

        self.nstack = nstack
        self.pre = nn.Sequential(
                    convolution(7, 3, 128, stride=2),
                    residual(3, 128, 256, stride=2))

        #-----------stack 1-----------------------------------------------
        self.kpr_stack1_1 = nn.ModuleList([nn.Sequential(
                    residual(3, 256, 256),
                    residual(3, 256, 256)) for _ in range(nstack)
                    ])
        self.kpd_stack1_1 = nn.ModuleList([nn.Sequential(
                    residual(3, 256, 256, stride=2),
                    residual(3, 256, 256)) for _ in range(nstack)
                    ])

        self.kpr_stack1_2 = nn.ModuleList([nn.Sequential(
                    residual(3, 256, 256),
                    residual(3, 256, 256)) for _ in range(nstack)
                    ])
        self.kpd_stack1_2 = nn.ModuleList([nn.Sequential(
                    residual(3, 256, 384, stride=2),
                    residual(3, 384, 384)) for _ in range(nstack)
                    ])
        
        self.kpr_stack1_3 = nn.ModuleList([nn.Sequential(
                    residual(3, 384, 384),
                    residual(3, 384, 384)) for _ in range(nstack)
                    ])
        self.kpd_stack1_3 = nn.ModuleList([nn.Sequential(
                    residual(3, 384, 384, stride=2),
                    residual(3, 384, 384)) for _ in range(nstack)
                    ])

        self.kpr_stack1_4 = nn.ModuleList([nn.Sequential(
                    residual(3, 384, 384),
                    residual(3, 384, 384)) for _ in range(nstack)
                    ])
        self.kpd_stack1_4 = nn.ModuleList([nn.Sequential(
                    residual(3, 384, 384, stride=2),
                    residual(3, 384, 384)) for _ in range(nstack)
                    ])

        # kpr_stack1[1] 要跟對應的數字的kpu[2]相加
        self.kpr_stack1_5 = nn.ModuleList([nn.Sequential(
                    residual(3, 384, 384),      
                    residual(3, 384, 384)) for _ in range(nstack)
                    ])
        self.kpd_stack1_5 = nn.ModuleList([nn.Sequential(
                    residual(3, 384, 512, stride=2),
                    residual(3, 512, 512)) for _ in range(nstack)
                    ])
        
        self.kp_stack1_middle = nn.ModuleList([nn.Sequential(
                    residual(3, 512, 512),
                    residual(3, 512, 512),
                    residual(3, 512, 512),
                    residual(3, 512, 512)) for _ in range(nstack)
                    ])

        self.kpu_stack1_5 = nn.ModuleList([nn.Sequential(
                    residual(3, 512, 512),
                    residual(3, 512, 384),
                    nn.Upsample(scale_factor=2)) for _ in range(nstack)
                    ])

        self.kpu_stack1_4 = nn.ModuleList([nn.Sequential(
                    residual(3, 384, 384),
                    residual(3, 384, 384),
                    nn.Upsample(scale_factor=2)) for _ in range(nstack)
                    ])

        self.kpu_stack1_3 = nn.ModuleList([nn.Sequential(
                    residual(3, 384, 384),
                    residual(3, 384, 384),
                    nn.Upsample(scale_factor=2)) for _ in range(nstack)
                    ])
        
        self.kpu_stack1_2 = nn.ModuleList([nn.Sequential(
                    residual(3, 384, 384),
                    residual(3, 384, 256),
                    nn.Upsample(scale_factor=2)) for _ in range(nstack)
                    ])
        
        self.kpu_stack1_1 = nn.ModuleList([nn.Sequential(
                    residual(3, 256, 256),
                    residual(3, 256, 256),
                    nn.Upsample(scale_factor=2)) for _ in range(nstack)
                    ])

        #-----------stack 2-----------------------------------------------
        self.kpr_stack2_1 = nn.ModuleList([
                    residual(3, 256, 256),
                    residual(3, 256, 256)
                    ])
        self.kpd_stack2_1 = nn.ModuleList([
                    residual(3, 256, 256, stride=2),
                    residual(3, 256, 256)
                    ])

        self.kpr_stack2_2 = nn.ModuleList([
                    residual(3, 256, 256),
                    residual(3, 256, 256)
                    ])
        self.kpd_stack2_2 = nn.ModuleList([
                    residual(3, 256, 384, stride=2),
                    residual(3, 384, 384)
                    ])
        
        self.kpr_stack2_3 = nn.ModuleList([
                    residual(3, 384, 384),
                    residual(3, 384, 384)
                    ])
        self.kpd_stack2_3 = nn.ModuleList([
                    residual(3, 384, 384, stride=2),
                    residual(3, 384, 384)
                    ])

        self.kpr_stack2_4 = nn.ModuleList([
                    residual(3, 384, 384),
                    residual(3, 384, 384)
                    ])
        self.kpd_stack2_4 = nn.ModuleList([
                    residual(3, 384, 384, stride=2),
                    residual(3, 384, 384)
                    ])

        # kpr_stack1[1] 要跟對應的數字的kpu[2]相加
        self.kpr_stack2_5 = nn.ModuleList([
                    residual(3, 384, 384),      
                    residual(3, 384, 384)
                    ])
        self.kpd_stack2_5 = nn.ModuleList([
                    residual(3, 384, 512, stride=2),
                    residual(3, 512, 512)
                    ])
        
        self.kp_stack2_middle = nn.ModuleList([
                    residual(3, 512, 512),
                    residual(3, 512, 512),
                    residual(3, 512, 512),
                    residual(3, 512, 512)
                    ])

        self.kpu_stack2_5 = nn.ModuleList([
                    residual(3, 512, 512),
                    residual(3, 512, 384),
                    nn.Upsample(scale_factor=2)
                    ])

        self.kpu_stack2_4 = nn.ModuleList([
                    residual(3, 384, 384),
                    residual(3, 384, 384),
                    nn.Upsample(scale_factor=2)
                    ])

        self.kpu_stack2_3 = nn.ModuleList([
                    residual(3, 384, 384),
                    residual(3, 384, 384),
                    nn.Upsample(scale_factor=2)
                    ])
        
        self.kpu_stack2_2 = nn.ModuleList([
                    residual(3, 384, 384),
                    residual(3, 384, 256),
                    nn.Upsample(scale_factor=2)
                    ])
        
        self.kpu_stack2_1 = nn.ModuleList([
                    residual(3, 256, 256),
                    residual(3, 256, 256),
                    nn.Upsample(scale_factor=2)
                    ])
        

        #--------------------------------------------------------------------
        # typical_conv = convolution
        self.cnvs = nn.ModuleList([
                        convolution(3, 256, 256) for _ in range(nstack)
                        ])

        self.inters_ = nn.ModuleList([
                            nn.Sequential(
                                nn.Conv2d(256, 256, (1, 1), bias=False),
                                nn.BatchNorm2d(256)) for _ in range(nstack)
                        ])
        
        self.relu = nn.ReLU(inplace=True)
        self.inters = residual(3, 256, 256)

    def forward(self, x):
        # print('image shape', image.shape)
        # inter = self.pre(image)
        # outs  = []
        for i in range(self.nstack):
            d11 = self.kpr_stack1_1[i](x)
            d12 = self.kpd_stack1_1[i](d11)
        
            d21 = self.kpr_stack1_2[i](d12)
            d22 = self.kpd_stack1_2[i](d21)

            d31 = self.kpr_stack1_3[i](d22)
            d32 = self.kpd_stack1_3[i](d31)

            d41 = self.kpr_stack1_4[i](d32)
            d42 = self.kpd_stack1_4[i](d41)

            d51 = self.kpr_stack1_5[i](d42)
            d52 = self.kpd_stack1_5[i](d51)

            h5m = self.kp_stack1_middle[i](d52)

            u51 = self.kpu_stack1_5[i](h5m)
            u52 = u51 + d51

            u41 = self.kpu_stack1_4[i](u52)
            u42 = u41 + d41

            u31 = self.kpu_stack1_3[i](u42)
            u32 = u31 + d31

            u21 = self.kpu_stack1_2[i](u32)
            u22 = u21 + d21

            u11 = self.kpu_stack1_1[i](u22)
            x = u11 + d11


        return x

class stack_hourglasses(nn.Module):
    def __init__(self, heads, nstack):
        super(stack_hourglasses, self).__init__()

        self.nstack = nstack

        self.pre = nn.Sequential(
                    convolution(7, 3, 128, stride=2),
                    residual(3, 128, 256, stride=2))

        # hourglass 設 1 來讓kp的hourglass一輪迭代一次
        self.kp = nn.ModuleList([
                    hourglasses(1) for _ in range(nstack)
                ])


        self.cnvs = nn.ModuleList([
                        convolution(3, 256, 256) for _ in range(nstack)
                        ])

        self.inters_ = nn.ModuleList([
                            nn.Sequential(
                                nn.Conv2d(256, 256, (1, 1), bias=False),
                                nn.BatchNorm2d(256)) for _ in range(nstack - 1)
                        ])
        self.cnvs2  = nn.ModuleList([
                        nn.Sequential(
                            nn.Conv2d(256, 256, (1, 1), bias=False),
                            nn.BatchNorm2d(256)
                        ) for _ in range(nstack - 1)
                        ])
        self.relu = nn.ReLU(inplace=True)
        self.inters = nn.ModuleList([
                        residual(3, 256, 256) for _ in range(nstack - 1)
                        ])
        
        # self.hm_conv = nn.ModuleList([
        #                     nn.Sequential(
        #                     convolution(3, 256, 256, with_bn = False),
        #                     nn.Conv2d(256, heads["hm"], (1, 1)))
        #                 ])
        self.hm_conv = nn.ModuleList([
                            convolution(3, 256, 256, with_bn = False),
                            nn.Conv2d(256, heads["hm"], (1, 1))
                        ])
        self.hm_conv[-1].bias.data.fill_(-2.19)

        self.wh_conv = nn.ModuleList([
                            nn.Sequential(
                            convolution(3, 256, 256, with_bn = False),
                            nn.Conv2d(256, heads["wh"], (1, 1)))
                        ])
        self.conv_k1_c1 = nn.Conv2d(heads["hm"], 1, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=2)
        self.channel_trans_conv = nn.Sequential(
                nn.Conv2d(256, 256 // 16, kernel_size=1),
                nn.LayerNorm([256 // 16, 1, 1]),
                nn.ReLU(inplace=False),
                nn.Conv2d(256 // 16, 256, kernel_size=1)
            )
    def obj_attention(self, feature, hm):
        f_batch, f_channel, f_height, f_width = feature.size()
        hm_batch, hm_channel, hm_height, hm_width = hm.size()
        # [N, C, H * W]
        feature = feature.view(f_batch, f_channel, f_height * f_width)
        # [N, 1, C, H * W]
        feature = feature.unsqueeze(1)
        # [N, 1, H, W]
        hm = self.conv_k1_c1(hm)
        # [N, 1, H * W]
        hm = hm.view(hm_batch, 1, hm_height * hm_width)
        # [N, 1, H * W]
        hm = self.softmax(hm)#softmax操作
        # [N, 1, H * W, 1]
        hm = hm.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(feature, hm)
        # [N, C, 1, 1]
        context = context.view(f_batch, f_channel, 1, 1)

        return context



    def forward(self, image):
        inter = self.pre(image)

        for i in range(self.nstack):
            kp = self.kp[i](inter)
            cnv = self.cnvs[i](kp)

            if i < self.nstack-1:
                inter = self.inters_[i](inter) + self.cnvs2[i](cnv)
                inter = self.relu(inter)
                inter = self.inters[i](inter)
            
        outs = []
        out = {}
        out["early"] = cnv
        hm = self.hm_conv[0](cnv)
        hm1 = self.hm_conv[1](hm)
        out["hm"] = hm1

        # object_attention
        context = self.obj_attention(hm, hm1.clone())
        trans = self.channel_trans_conv(context)
        # x = x + trans
        hm = hm * trans

        wh = self.wh_conv[0](hm)
        out["wh"] = wh

        outs.append(out)

        return outs

def get_large_hourglass_net(num_class):
    heads = {'hm': num_class,
                'wh': 2,
                'reg': 2}
    
    model = stack_hourglasses(heads, 2)
    return model


#######################################################################
# save model
def save_model(path, epoch, model, optimizer=None):
    state_dict = model.state_dict()
    data = {'epoch': epoch,
          'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()

    torch.save(data, path)
    ############################
    # torch.save(model, path)

# load model
def load_model(path, model, device):
    data = torch.load(path, map_location=device)
    print(data.keys())
    model.load_state_dict(data["state_dict"], strict = True)
    return model
    # model = torch.load(path)
    # return model