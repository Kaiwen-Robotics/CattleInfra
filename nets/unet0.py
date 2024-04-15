import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.resnet import resnet50
from nets.vgg import VGG16
from einops import rearrange

class Cluster(nn.Module):
    def __init__(self, dim, out_dim, proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24,
                 return_center=False):
        """

        :param dim:  channel nubmer
        :param out_dim: channel nubmer
        :param proposal_w: the sqrt(proposals) value, we can also set a different value
        :param proposal_h: the sqrt(proposals) value, we can also set a different value
        :param fold_w: the sqrt(number of regions) value, we can also set a different value
        :param fold_h: the sqrt(number of regions) value, we can also set a different value
        :param heads:  heads number in context cluster
        :param head_dim: dimension of each head in context cluster
        :param return_center: if just return centers instead of dispatching back (deprecated).
        """
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.f = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for similarity
        self.proj = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)  # for projecting channel number
        self.v = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for value. 没有指定padding和stride，所以padding=0，stride=1
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_w, proposal_h))  # average pooling layer, 分别改变输入的w和h为proposal_w和proposal_h
        self.fold_w = fold_w
        self.fold_h = fold_h
        self.return_center = return_center

    def forward(self, x):  # [b,c,w,h]
        value = self.v(x)   # [b,c,h,w]->[b,heads*head_dim,h,w], 1*1 conv
        x = self.f(x)   # [b,c,h,w]->[b,heads*head_dim,h,w], 1*1 conv
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)    # [b,heads*head_dim,h,w]->[b*heads,head_dim,h,w]
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)    # [b,heads*head_dim,h,w]->[b*heads,head_dim,h,w]
        if self.fold_w > 1 and self.fold_h > 1:
            # split the big feature maps to small local regions to reduce computations.
            b0, c0, w0, h0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,    # [b*heads,head_dim,h,w]->[b*heads*self.fold_w*self.fold_h,head_dim,h/self.fold_h,w/self.fold_w]
                          f2=self.fold_h)  # [bs*blocks,c,ks[0],ks[1]]
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)    # [b*heads,head_dim,h,w]->[b*heads*self.fold_w*self.fold_h,head_dim,h/self.fold_h,w/self.fold_w]
        b, c, w, h = x.shape
        centers = self.centers_proposal(x)  # [b,c,C_W,C_H], we set M = C_W*C_H and N = w*h。这里的C_H*C_W个c维向量都是center
        value_centers = rearrange(self.centers_proposal(value), 'b c w h -> b (w h) c')  # [b,C_W,C_H,c]
        b, c, ww, hh = centers.shape
        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )  # [B,M,N]
        # we use mask to sololy assign each point to one center
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)     # [B,1,N], M个center、N个点, 存放和每个点最相似的center的sim值
        mask = torch.zeros_like(sim)  # binary #[B,M,N]
        mask.scatter_(1, sim_max_idx, 1.)   # mask是和sim同shape的tensor，sim在dim=1的维度上对应在mask中的值为1.0，其他的值为0
        sim = sim * mask    # 只保留每个点和对应的center之间的最大sim值，其他sim值被置零，[B,M,N]
        value2 = rearrange(value, 'b c w h -> b (w h) c')  # [B,N,D]

        """
        到这一行之前, 我们有sim (shape: B,M,N); value2 (shape: B,N,D)。一共N个点, M个center, embedding dim为D。
        input x通过两个1*1 conv, 分别生成x和value; sim由x计算所得, value2由value reshape所得(大体是这样)
        sim中存储了每个点属于哪个center(簇), 并存储点和其所属center的similarity。
        """

        # aggregate step, out shape [B,M,D]
        # Feature Aggregating in paper 3.2
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (
                    mask.sum(dim=-1, keepdim=True) + 1.0)  # [B,M,D]

        # 如果return_center为True就返回center的值; 反之返回经过Feature Dispatching之后的Feature maps的值
        if self.return_center:
            out = rearrange(out, "b (w h) c -> b c w h", w=ww)
        else:
            # dispatch step, return to each point in a cluster
            # Feature Dispatching in paper 3.2
            out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)  # [B,N,D]
            out = rearrange(out, "b (w h) c -> b c w h", w=w)

        if self.fold_w > 1 and self.fold_h > 1:
            # recover the splited regions back to big feature maps if use the region partition.
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.proj(out)
        return out

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        self.cluster4 = Cluster(out_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        self.cluster3 = Cluster(out_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.cluster2 = Cluster(out_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        self.cluster1 = Cluster(out_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up4 = self.cluster4(up4)
        up3 = self.up_concat3(feat3, up4)
        up3 = self.cluster3(up3)
        up2 = self.up_concat2(feat2, up3)
        up2 = self.cluster2(up2)
        up1 = self.up_concat1(feat1, up2)
        up1 = self.cluster1(up1)
        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True

# 先对最后一维normalize再matmul，实现cosine similarity matrix的计算
def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim