"""DenseHARNet: multi-scale inception 1D conv (k=1,3,5,11) + SE + dilated temporal stack."""
import torch, torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc=nn.Sequential(nn.Linear(ch,ch//r), nn.ReLU(), nn.Linear(ch//r,ch), nn.Sigmoid())
    def forward(self,x):
        w=self.fc(x.mean(dim=2)).unsqueeze(-1)
        return x*w

class InceptionBlock(nn.Module):
    def __init__(self, in_ch, branch_ch=16, out_ch=48):
        super().__init__()
        self.branches=nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_ch, branch_ch, k, padding=k//2), nn.GroupNorm(4,branch_ch), nn.ReLU())
            for k in (1,3,5,11)])
        self.proj=nn.Sequential(nn.Conv1d(branch_ch*4, out_ch, 1), nn.GroupNorm(8,out_ch), nn.ReLU())
        self.se=SEBlock(out_ch)
        self.short=nn.Conv1d(in_ch, out_ch, 1) if in_ch!=out_ch else nn.Identity()
    def forward(self,x):
        y=self.proj(torch.cat([b(x) for b in self.branches], dim=1))
        return torch.relu(self.se(y)+self.short(x))

class TemporalBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv1d(ch,ch,3,padding=dilation,dilation=dilation),
                                nn.GroupNorm(8,ch), nn.ReLU())
    def forward(self,x): return torch.relu(x+self.conv(x))

class DenseHARNet(nn.Module):
    def __init__(self, in_ch=6, num_classes=6, width=48, branch_ch=16):
        super().__init__()
        self.width=width
        self.stem=nn.Sequential(nn.Conv1d(in_ch,width,5,padding=2,stride=2), nn.GroupNorm(8,width), nn.ReLU())
        self.inc1=InceptionBlock(width,branch_ch,width)
        self.inc2=InceptionBlock(width,branch_ch,width)
        self.temp=nn.Sequential(TemporalBlock(width,1), TemporalBlock(width,2), TemporalBlock(width,4), TemporalBlock(width,8))
        self.head=nn.Conv1d(width,num_classes,1)
    def forward(self,x):  # (B,6,L) -> (B,L,6); encoder runs at half rate, head upsampled x2
        L=x.shape[-1]
        x=self.temp(self.inc2(self.inc1(self.stem(x))))
        x=torch.nn.functional.interpolate(x, size=L, mode="linear", align_corners=False)
        return self.head(x).transpose(1,2)

class WindowClassifier(nn.Module):
    """Baseline: same inception backbone, one label per 128-length window."""
    def __init__(self, in_ch=6, num_classes=6, width=48, branch_ch=16):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv1d(in_ch,width,5,padding=2,stride=2), nn.GroupNorm(8,width), nn.ReLU())
        self.inc1=InceptionBlock(width,branch_ch,width)
        self.inc2=InceptionBlock(width,branch_ch,width)
        self.head=nn.Linear(width,num_classes)
    def forward(self,x):
        x=self.inc2(self.inc1(self.stem(x)))
        return self.head(x.mean(dim=2))

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

if __name__=="__main__":
    for w in (32,48,64):
        m=DenseHARNet(width=w); print(f"DenseHARNet width={w} params: {count_params(m):,}")
    print(DenseHARNet()(torch.randn(2,6,512)).shape)
    b=WindowClassifier(); print(f"WindowClassifier params: {count_params(b):,}"); print(b(torch.randn(2,6,128)).shape)
