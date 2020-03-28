import torch
import torch.nn as nn

'''
仅使用embed结构
'''


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size//2), bias=bias)
class FAFusion(nn.Module):
    def __init__(self,in_dim,out_dim):
        #indim是outdim的整数倍
        super(FAFusion,self).__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.reduction=self.in_dim//self.out_dim
        self.ca=nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.in_dim,self.in_dim//16,1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_dim//16, self.in_dim, 1, padding=0, bias=True),
            nn.Sigmoid()
            ])
        self.pa=nn.Sequential(
            nn.Conv2d(out_dim,out_dim//8,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim//8,1,3,padding=1),
            nn.Sigmoid()
        )
    def forward(self,x):
        #对应临界两个fusion，不使用论文的conv,使用FAfusion
        # w=self.ca(torch.cat([x1,x2],dim=1))
        # w=w.view(-1,2,self.out_dim)[:,:,:,None,None]
        # out=w[:,0,::]*x1+w[:,1,::]*x2
        # x=torch.cat([x1,x2],dim=1)
        w=self.ca(x)
        out=w*x
        out=out.view(-1,self.reduction,self.out_dim,x.size(-2),x.size(-1))
        out=torch.sum(out,dim=1)
        att=self.pa(out)
        out=att*out
        return out
        
class FALayer(nn.Module):
    def __init__(self,channels):
        super(FALayer,self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.ca=nn.Sequential(
            nn.Conv2d(channels,channels//8,1,padding=0,bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//8,channels,1,padding=0,bias=True),
            nn.Sigmoid())
        self.pa=nn.Sequential(
            nn.Conv2d(channels,channels//8,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//8,1,3,padding=1),
            nn.Sigmoid()
        )
    def forward(self,x):
        y=self.avg_pool(x)
        y=self.ca(y)
        x=x*y
        y=self.pa(x)
        return x*y

class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size,):
        super(Block, self).__init__()
        self.conv1=conv(dim, dim, kernel_size, bias=True)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=conv(dim,dim,kernel_size,bias=True)
        self.falyer=FALayer(dim)
    def forward(self, x):
        res=self.act1(self.conv1(x))
        res=res+x #local res
        res=self.conv2(res)
        res=self.falyer(res)
        res += x 
        return res
class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [ Block(conv, dim, kernel_size)  for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)
    def forward(self, x):
        res = self.gp(x)
        res += x
        return res

class BRM(nn.Module):
    def __init__(self, channels, blocks,bp=True):
        super(BRM, self).__init__()
        self.bp=bp
        self.flow = Group(default_conv,channels,kernel_size=3,blocks=blocks)
    def forward(self, x):
        if self.bp:
            out=self.flow(x)
            return out,out
        else :
            out = self.flow(x)
            return out

class Embed_FFA(nn.Module):
    def __init__(self,brms=3,blocks=20,channels=64):
        super(Embed_FFA, self).__init__()
        self.n_brms = brms
        self.channels = channels
        self.blocks=blocks
        kernel_size = 3
        # feature shallow extraction
        self.head = nn.Sequential(*[
            nn.Conv2d(3, channels * 4, kernel_size, padding=kernel_size // 2),
            nn.PReLU(channels * 4),
            nn.Conv2d(channels * 4, channels, kernel_size, padding=kernel_size // 2),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.PReLU(channels)
        ])

        #fusion layer
        self.fusion_convs = nn.ModuleList()
        for i in range(self.n_brms - 1):
            self.fusion_convs.append(FAFusion(in_dim=channels*2,out_dim=channels))#2个一融合

        # embedded block residual learning
        self.brms = nn.ModuleList()
        for i in range(self.n_brms - 1):
            self.brms.append(BRM(channels,self.blocks,True))
        self.brms.append(BRM(channels,self.blocks,False))
        # Fusion reconstruction
        self.tail = nn.Sequential(*[
            FAFusion(in_dim=self.n_brms*self.channels,out_dim=self.channels),
            default_conv(self.channels, self.channels, kernel_size),
            default_conv(self.channels, 3, kernel_size)])

    def forward(self, x1):
        x = self.head(x1)
        out = []
        sr_sets = []
        # 前面的self.n_brms-1层
        for i in range(self.n_brms - 1):
            x, sr = self.brms[i](x)
            sr_sets.append(sr)
        # 最后的第n_brms层
        sr = self.brms[self.n_brms - 1](x)
        out.append(sr)
        for i in range(self.n_brms - 1):#0,1
            # sr = sr + sr_sets[self.n_brms - i - 2]
            sr = self.fusion_convs[i](torch.cat([sr,sr_sets[self.n_brms-i-2]],dim=1))
            out.append(sr)
        x = self.tail(torch.cat(out, dim=1))
        return x+x1

if __name__ == "__main__":
    net=Embed_FFA(brms=3,blocks=20)
    print('# Net parameters:', sum(param.numel() for param in net.parameters()))
    print('done')