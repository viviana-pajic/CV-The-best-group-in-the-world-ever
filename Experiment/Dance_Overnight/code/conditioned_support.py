"""Give the crop network an explicit target-box mask in crowded crops."""
import sys
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'Dance_Floor_Contact'))
from support_train import SupportNet,SIZE

class ConditionedSupportNet(SupportNet):
    def __init__(self,pretrained=True):
        super().__init__(pretrained);old=self.stem[0];conv=nn.Conv2d(4,64,7,stride=2,padding=3,bias=False)
        with torch.no_grad():conv.weight[:,:3].copy_(old.weight);conv.weight[:,3:].zero_()
        self.stem[0]=conv

class AttentionSupportNet(SupportNet):
    """Assess reportability using local contact evidence as well as whole-person context."""
    def __init__(self,pretrained=True,condition=True):
        super().__init__(pretrained)
        if condition:
            old=self.stem[0];conv=nn.Conv2d(4,64,7,stride=2,padding=3,bias=False)
            with torch.no_grad():conv.weight[:,:3].copy_(old.weight);conv.weight[:,3:].zero_()
            self.stem[0]=conv
        self.visible=nn.Sequential(nn.Linear(641,128),nn.ReLU(),nn.Dropout(.15),nn.Linear(128,1))
    def forward(self,x):
        a=self.layer2(self.stem(x));b=self.layer3(a);features=torch.cat([a,F.interpolate(b,size=a.shape[-2:],mode='bilinear',align_corners=False)],1);hm=self.heat(features)[:,0]
        attention=hm.detach().flatten(1).softmax(1);local=(features.flatten(2)*attention[:,None]).sum(2);entropy=-(attention*attention.clamp_min(1e-9).log()).sum(1,keepdim=True)/np.log(attention.shape[1]);visible=self.visible(torch.cat([b.mean((2,3)),local,entropy],1))[:,0]
        return F.interpolate(hm[:,None],scale_factor=2,mode='bilinear',align_corners=False)[:,0],visible

def target_mask(box,r,flip=False):
    x1,y1,x2,y2=np.asarray(box,float);l,t,rr,b=r
    xx=l+(np.arange(SIZE[0])+.5)/SIZE[0]*(rr-l);yy=t+(np.arange(SIZE[1])+.5)/SIZE[1]*(b-t)
    m=((yy[:,None]>=y1)&(yy[:,None]<=y2)&(xx[None]>=x1)&(xx[None]<=x2)).astype(np.float32)
    if flip:m=m[:,::-1].copy()
    return torch.from_numpy(m[None])


class TemporalSupportNet(AttentionSupportNet):
    """Three aligned image crops; initialize with current-frame evidence only."""
    def __init__(self,checkpoint=None):
        super().__init__(False,False)
        if checkpoint is not None:self.load_state_dict(checkpoint)
        old=self.stem[0];conv=nn.Conv2d(9,64,7,stride=2,padding=3,bias=False)
        with torch.no_grad():
            conv.weight.zero_();conv.weight[:,3:6].copy_(old.weight)
        self.stem[0]=conv
