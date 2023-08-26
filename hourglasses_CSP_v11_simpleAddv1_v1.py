# ------------------------------------------------------------------------------
# This code is base on 
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from dcn import DeformableConv2d

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

class CSP_Block(nn.Module):

    def __init__(self, in_channels, part2_down = True, part_ratio=0.5):
        super(CSP_Block, self).__init__()
        self.part1_chnls = int(in_channels * part_ratio)
        self.part2_chnls = in_channels - self.part1_chnls
        self.part2_down = part2_down
        # self.conv_down_block = nn.ModuleList([nn.Sequential(
        #                         nn.Conv2d(self.part2_chnls, 2 * self.part2_chnls, (1, 1), stride=(1, 1), bias=False),
        #                         nn.BatchNorm2d(2 * self.part2_chnls),
        #                         convolution(3, 2 * self.part2_chnls, 2 * self.part2_chnls),
        #                         nn.Conv2d(2 * self.part2_chnls, self.part2_chnls, (1, 1), stride=(1, 1), bias=False),
        #                         nn.BatchNorm2d(self.part2_chnls))
        #                     ])
        self.maxpool = nn.MaxPool2d((2,2), stride=(2,2))
        self.transition_1 = convolution(1, self.part2_chnls, self.part2_chnls)
        # self.transition_2 = convolution(1, self.part2_chnls, self.part2_chnls)
        self.avg_pool_1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_2 = nn.AdaptiveAvgPool2d(1)
        self.sigmoid_1 = nn.Sigmoid()
        self.sigmoid_2 = nn.Sigmoid()
        self.spatial_sigmoid = nn.Sigmoid()
        # self.relu_part1  = nn.ReLU(inplace=True)
        # self.res_down_block = nn.ModuleList([nn.Sequential(
        #                         residual(3, self.part2_chnls, self.part2_chnls, stride=2),
        #                         nn.MaxPool2d(2, stride=2))
        #                     ])
        # trans_chnls = self.part2_chnls + k * num_layers
        # self.transtion = BN_Conv2d(trans_chnls, trans_chnls, 1, 1, 0)

    def forward(self, x):
        part1 = x[:, :self.part1_chnls, :, :]
        down_part2 = x[:, self.part1_chnls:, :, :]
        origin_total = part1 + down_part2

        # part1 = self.transition_1(part1)
        sig_part1 = self.avg_pool_1(part1)
        sig_part1 = self.sigmoid_1(sig_part1)
        part1 = part1 * (1 - sig_part1)

        # trans_part2 = self.transition_2(down_part2)
        sig_part2 = self.avg_pool_2(down_part2)
        sig_part2 = self.sigmoid_2(sig_part2)
        trans_part2 = down_part2 * (1 - sig_part2)

        part1 = part1 + trans_part2
        res_part1 = self.transition_1(part1)
        res_part1 = self.spatial_sigmoid(res_part1)
        part1 = part1 * res_part1

        origin_total = origin_total * (1-res_part1)
        part1 = part1 + origin_total

        if self.part2_down:
            # down_part2 = part1 + down_part2
            down_part2 = self.maxpool(down_part2)
        else:
            down_part2 = torch.cat((part1, down_part2), dim=1)
            
        # part2 = self.transtion(part2)
        
        return part1, down_part2


class stack_hourglasses(nn.Module):
    def __init__(self, heads, nstack):
        super(stack_hourglasses, self).__init__()
        self.nstack = nstack

        self.pre = nn.Sequential(
                    convolution(7, 3, 128, stride=2),
                    residual(3, 128, 256, stride=2))
        
        # self.pre_deform = nn.Sequential(
        #             DeformableConv2d(in_channels=3, out_channels=128, kernel_size=7, stride=2, padding=1),
        #             DeformableConv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=1))


        # --------------------------------------------------
        self.csp_0 = CSP_Block(256)
        self.kpr_stack1_1 = nn.ModuleList([nn.Sequential(
                    residual(3, 256, 256),
                    residual(3, 256, 256)) for _ in range(nstack)
                    ])
        self.kpd_stack1_1 = nn.ModuleList([nn.Sequential(
                    residual(3, 256, 256, stride=2),
                    residual(3, 256, 256)) for _ in range(nstack)
                    ])
        self.csp_1 = CSP_Block(256)

        self.kpr_stack1_2 = nn.ModuleList([nn.Sequential(
                    residual(3, 256, 256),
                    residual(3, 256, 256)) for _ in range(nstack)
                    ])
        self.kpd_stack1_2 = nn.ModuleList([nn.Sequential(
                    residual(3, 256, 384, stride=2),
                    residual(3, 384, 384)) for _ in range(nstack)
                    ])
        self.csp_2 = CSP_Block(384)

        self.kpr_stack1_3 = nn.ModuleList([nn.Sequential(
                    residual(3, 384, 384),
                    residual(3, 384, 384)) for _ in range(nstack)
                    ])
        self.kpd_stack1_3 = nn.ModuleList([nn.Sequential(
                    residual(3, 384, 384, stride=2),
                    residual(3, 384, 384)) for _ in range(nstack)
                    ])
        self.csp_3 = CSP_Block(384)

        self.kpr_stack1_4 = nn.ModuleList([nn.Sequential(
                    residual(3, 384, 384),
                    residual(3, 384, 384)) for _ in range(nstack)
                    ])
        self.kpd_stack1_4 = nn.ModuleList([nn.Sequential(
                    residual(3, 384, 384, stride=2),
                    residual(3, 384, 384)) for _ in range(nstack)
                    ])
        self.csp_4 = CSP_Block(384, part2_down = False)

        # kpr_stack1[1] 要跟對應的數字的kpu[2]相加
        # self.kpr_stack1_5 = nn.ModuleList([nn.Sequential(
        #             residual(3, 384, 384),      
        #             residual(3, 384, 384)) for _ in range(nstack)
        #             ])
        # self.kpd_stack1_5 = nn.ModuleList([nn.Sequential(
        #             residual(3, 384, 512, stride=2),
        #             residual(3, 512, 512)) for _ in range(nstack)
        #             ])
        
        # self.kp_stack1_middle = nn.ModuleList([nn.Sequential(
        #             residual(3, 512, 512),
        #             residual(3, 512, 512),
        #             residual(3, 512, 512),
        #             residual(3, 512, 512)) for _ in range(nstack)
        #             ])

        # self.kpu_stack1_5 = nn.ModuleList([nn.Sequential(
        #             residual(3, 512, 512),
        #             residual(3, 512, 384),
        #             nn.Upsample(scale_factor=2)) for _ in range(nstack)
        #             ])

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
        self.feat_up = nn.ModuleList([
                        convolution(3, 384, 384)
                        ])
        # self.hm_conv = nn.ModuleList([
        #                     nn.Sequential(
        #                     convolution(3, 256, 256, with_bn = False),
        #                     nn.Conv2d(256, heads["hm"], (1, 1)))
        #                 ])
        self.hm_conv = nn.ModuleList([
                            convolution(3, 384, 384, with_bn = False),
                            nn.Conv2d(384, heads["hm"], (1, 1))
                        ])
        self.hm_conv[-1].bias.data.fill_(-2.19)

        self.wh_conv = nn.ModuleList([
                            nn.Sequential(
                            convolution(3, 384, 384, with_bn = False),
                            nn.Conv2d(384, heads["wh"], (1, 1)))
                        ])
        self.reg_conv = nn.ModuleList([
                            nn.Sequential(
                            convolution(3, 384, 384, with_bn = False),
                            nn.Conv2d(384, heads["reg"], (1, 1)))
                        ])
        # --------- CSP up -----------------
        self.kpu_stack2_4 = nn.ModuleList([nn.Sequential(
                                residual(3, 384, 384),
                                residual(3, 384, 384))
                                ])
        self.kpu_stack2_3 = nn.ModuleList([nn.Sequential(
                                # residual(3, 768, 768),
                                residual(3, 768, 768),
                                residual(3, 768, 384))
                                ])
        self.up3 =  nn.Upsample(scale_factor=2)
        # self.kpu_stack2_3_skip = nn.ModuleList([nn.Sequential(
        #                         convolution(1, 384, 192))
        #                         ])
                               
        self.kpu_stack2_2 = nn.ModuleList([nn.Sequential(
                                # residual(3, 768, 768),
                                residual(3, 768, 768),
                                residual(3, 768, 384))
                                ])
        self.up2 =  nn.Upsample(scale_factor=2)
        # self.kpu_stack2_2_skip = nn.ModuleList([nn.Sequential(
        #                         convolution(1, 384, 192))
        #                         ])

        self.kpu_stack2_1 = nn.ModuleList([nn.Sequential(
                                # residual(3, 640, 640),
                                residual(3, 704, 704),
                                residual(3, 704, 256))
                                ])
        self.up1 =  nn.Upsample(scale_factor=2)
        # self.kpu_stack2_1_skip = nn.ModuleList([nn.Sequential(
        #                         convolution(1, 256, 128))
        #                         ])
        
        self.kpu_stack2_0 = nn.ModuleList([nn.Sequential(
                                # residual(3, 512, 512),
                                residual(3, 512, 512),
                                residual(3, 512, 256))
                                ])
        self.up0 =  nn.Upsample(scale_factor=2)
        # self.kpu_stack2_0_skip = nn.ModuleList([nn.Sequential(
        #                         convolution(1, 256, 128))
        #                         ])


        # object aware
        self.conv_k1_c1 = nn.Conv2d(heads["hm"], 1, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=2)
        # self.sigmoid = nn.Sigmoid()
        self.channel_trans_conv = nn.Sequential(
                nn.Conv2d(384, 384 // 16, kernel_size=1),
                nn.LayerNorm([384 // 16, 1, 1]),
                nn.ReLU(inplace=False),
                nn.Conv2d(384 // 16, 384, kernel_size=1)
            )

    def simple_addition(self, feature, hm):
        f_batch, f_channel, f_height, f_width = feature.size()
        hm_batch, hm_channel, hm_height, hm_width = hm.size()
        # [N, C, H * W]
        feature = feature.view(f_batch, f_channel, f_height * f_width)
        # [N, 1, C, H * W]
        feature = feature.unsqueeze(1)
        # [6,128,128]
        # hm = torch.sum(hm, dim = 1)
        temp = (hm[:,0,:,:] + 2*hm[:,1,:,:] + 2*hm[:,2,:,:] + hm[:,3,:,:] + hm[:,4,:,:] + hm[:,5,:,:])/8
        # [6,1,128,128]
        temp = temp.unsqueeze(1)
        # [N, 1, H * W]
        temp = temp.view(hm_batch, 1, hm_height * hm_width)
        # [N, 1, H * W]
        temp = self.softmax(temp)#softmax操作
        # [N, 1, H * W, 1]
        temp = temp.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(feature, temp)
        # [N, C, 1, 1]
        context = context.view(f_batch, f_channel, 1, 1)

        return context

    def forward(self, image):
        inter = self.pre(image)

        # for i in range(self.nstack):
        #     kp = self.kp[i](inter)
        #     cnv = self.cnvs[i](kp)

        #     if i < self.nstack-1:
        #         inter = self.inters_[i](inter) + self.cnvs2[i](cnv)
        #         inter = self.relu(inter)
        #         inter = self.inters[i](inter)
        cat_csp0, down_csp0 = self.csp_0(inter)
        #--------- stack kp 1
        
        d11 = self.kpr_stack1_1[0](inter)
        d12 = self.kpd_stack1_1[0](d11)
        cat_csp1, down_csp1 = self.csp_1(d12)

        d21 = self.kpr_stack1_2[0](d12)
        d22 = self.kpd_stack1_2[0](d21)
        cat_csp2, down_csp2 = self.csp_2(d22)

        d31 = self.kpr_stack1_3[0](d22)
        d32 = self.kpd_stack1_3[0](d31)
        cat_csp3, down_csp3 = self.csp_3(d32)

        d41 = self.kpr_stack1_4[0](d32)
        d42 = self.kpd_stack1_4[0](d41)
        cat_csp4, down_csp4 = self.csp_4(d42)

        # d51 = self.kpr_stack1_5[0](d42)
        # d52 = self.kpd_stack1_5[0](d51)

        # h5m = self.kp_stack1_middle[0](d52)

        # u51 = self.kpu_stack1_5[0](h5m)
        # u52 = u51 + d51

        u41 = self.kpu_stack1_4[0](d42)
        u42 = u41 + d41

        u31 = self.kpu_stack1_3[0](u42)
        u32 = u31 + d31

        u21 = self.kpu_stack1_2[0](u32)
        u22 = u21 + d21

        u11 = self.kpu_stack1_1[0](u22)
        kp = u11 + d11

        # ---- bridge
        cnv = self.cnvs[0](kp)
        inter = self.inters_[0](inter) + self.cnvs2[0](cnv)
        inter = self.relu(inter)
        inter = self.inters[0](inter)

        #--------- stack kp 2
        d11 = self.kpr_stack1_1[1](inter)
        d12 = self.kpd_stack1_1[1](d11)
    
        d21 = self.kpr_stack1_2[1](d12)
        d22 = self.kpd_stack1_2[1](d21)

        d31 = self.kpr_stack1_3[1](d22)
        d32 = self.kpd_stack1_3[1](d31)

        d41 = self.kpr_stack1_4[1](d32)
        d42 = self.kpd_stack1_4[1](d41)

        # d51 = self.kpr_stack1_5[1](d42)
        # d52 = self.kpd_stack1_5[1](d51)

        # h5m = self.kp_stack1_middle[1](d52)

        # u51 = self.kpu_stack1_5[1](h5m)
        # u52 = u51 + d51
        
        d42 = self.kpu_stack2_4[0](d42)
        csp4 = d42 + down_csp4
        pass4 = torch.cat((down_csp3, cat_csp4), dim=1)
        csp4 = torch.cat((pass4, csp4), dim=1)

        csp3 = self.kpu_stack2_3[0](csp4)
        csp3 = self.up3(csp3) 
        # d41 = self.kpu_stack2_3_skip[0](d41)
        csp3 = d41 + csp3
        pass3 = torch.cat((down_csp2, cat_csp3), dim=1)
        csp3 = torch.cat((pass3, csp3), dim=1)

        csp2 = self.kpu_stack2_2[0](csp3)
        csp2 = self.up2(csp2)
        # d31 = self.kpu_stack2_2_skip[0](d31)
        csp2 = d31 + csp2
        pass2 = torch.cat((down_csp1, cat_csp2), dim=1)
        csp2 = torch.cat((pass2, csp2), dim=1)

        csp1 = self.kpu_stack2_1[0](csp2)
        csp1 = self.up1(csp1)
        # d21 = self.kpu_stack2_1_skip[0](d21)
        csp1 = d21 + csp1
        pass1 = torch.cat((down_csp0, cat_csp1), dim=1)
        csp1 = torch.cat((pass1, csp1), dim=1)

        csp0 = self.kpu_stack2_0[0](csp1)
        csp0 = self.up1(csp0)
        # d11 = self.kpu_stack2_0_skip[0](d11)
        csp0 = d11 + csp0
        csp0 = torch.cat((cat_csp0, csp0), dim=1)


        # cnv = self.cnvs[1](csp0)
        cnv = self.feat_up[0](csp0)


        outs = []
        out = {}
        # out["early"] = cnv
        hm = self.hm_conv[0](cnv)
        hm1 = self.hm_conv[1](hm)
        out["hm"] = hm1



        # object_attention
        # context = self.simple_addition(hm, hm1.clone())
        context = self.simple_addition(hm, hm1.clone().detach())
        trans = self.channel_trans_conv(context)
        # x = x + trans
        # hm = hm * trans
        att = hm + trans


        wh = self.wh_conv[0](att)
        out["wh"] = wh

        reg = self.reg_conv[0](att)
        out["reg"] = reg

        outs.append(out)

        


        return outs

def get_large_hourglass_net(num_class):
    heads = {'hm': num_class,
                'wh': 2,
                'reg': 2}
    
    model = stack_hourglasses(heads, 2)
    return model