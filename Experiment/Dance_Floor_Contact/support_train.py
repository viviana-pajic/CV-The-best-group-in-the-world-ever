"""Train a crop-based support heatmap and contact-reportability classifier."""
import os,json,time,random,hashlib
os.environ.setdefault('OMP_NUM_THREADS','4')
import cv2,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision.models import resnet18,ResNet18_Weights
from support_label_setup import OUT,TRAIN,VAL,TEST
from corrected_setup import read

SIZE=(160,224) # width,height
MEAN=np.array([.485,.456,.406],np.float32)[:,None,None]
STD=np.array([.229,.224,.225],np.float32)[:,None,None]

def roi(box,jitter=False):
    x1,y1,x2,y2=map(float,box);w=max(24,x2-x1);h=max(30,y2-y1)
    cx=(x1+x2)/2;cy=(y1+y2)/2
    if jitter:cx+=random.uniform(-.08,.08)*w;cy+=random.uniform(-.06,.06)*h;w*=random.uniform(.9,1.15);h*=random.uniform(.9,1.15)
    return np.array([max(0,cx-w*.70),max(0,cy-h*.62),min(1280,cx+w*.70),min(720,cy+h*.72)],np.float32)

def crop(im,r):
    x1,y1,x2,y2=r
    matrix=np.array([[SIZE[0]/(x2-x1),0,-x1*SIZE[0]/(x2-x1)],[0,SIZE[1]/(y2-y1),-y1*SIZE[1]/(y2-y1)]],np.float32)
    return cv2.warpAffine(im,matrix,SIZE,flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

def tensor(im):return torch.from_numpy(((im[:,:,::-1].transpose(2,0,1).astype(np.float32)/255)-MEAN)/STD)

class Samples(Dataset):
    def __init__(self,frames,augment=False):
        labels=read(OUT/'labels.json');self.images={k:cv2.imread(str(OUT/'images'/f'{k:03}.png')) for k in frames};self.rows=[(k,o) for k in frames for o in labels[str(k)] if o['box'] is not None];self.augment=augment
    def __len__(self):return len(self.rows)*(4 if self.augment else 1)
    def __getitem__(self,i):
        k,o=self.rows[i%len(self.rows)];r=roi(o['box'],self.augment);im=crop(self.images[k],r)
        vis=float(o['floor'] is not None);p=(np.array(o['floor'],np.float32)-r[:2])/(r[2:]-r[:2]) if vis else np.array([.5,.5],np.float32)
        if vis and (p.min()<=0 or p.max()>=1):
            r=roi(o['box']);im=crop(self.images[k],r);p=(np.array(o['floor'],np.float32)-r[:2])/(r[2:]-r[:2])
        if self.augment:
            if random.random()<.5:im=im[:,::-1].copy();p[0]=1-p[0]
            im=np.clip(im.astype(np.float32)*random.uniform(.8,1.2)+random.uniform(-12,12),0,255).astype(np.uint8)
        assert not vis or (p.min()>0 and p.max()<1),(k,o['id'],p)
        return tensor(im),torch.tensor(p,dtype=torch.float32),torch.tensor(vis),torch.tensor(r)

class SupportNet(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__();m=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem=nn.Sequential(m.conv1,m.bn1,m.relu,m.maxpool,m.layer1);self.layer2=m.layer2;self.layer3=m.layer3
        self.heat=nn.Sequential(nn.Conv2d(384,128,3,padding=1),nn.ReLU(),nn.Conv2d(128,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,1,1))
        self.visible=nn.Sequential(nn.Linear(256,64),nn.ReLU(),nn.Dropout(.15),nn.Linear(64,1))
        for p in self.stem.parameters():p.requires_grad=False
    def train(self,mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m,nn.BatchNorm2d):m.eval()
        return self
    def forward(self,x):
        a=self.layer2(self.stem(x));b=self.layer3(a)
        hm=self.heat(torch.cat([a,F.interpolate(b,size=a.shape[-2:],mode='bilinear',align_corners=False)],1))
        hm=F.interpolate(hm,scale_factor=2,mode='bilinear',align_corners=False)[:,0]
        return hm,self.visible(b.mean((2,3)))[:,0]

def coordinates(hm):
    n,h,w=hm.shape;prob=hm.flatten(1).softmax(1).reshape(n,h,w)
    xx=(torch.arange(w,device=hm.device)+.5)/w;yy=(torch.arange(h,device=hm.device)+.5)/h
    return torch.stack([(prob.sum(1)*xx).sum(1),(prob.sum(2)*yy).sum(1)],1)

def loss(hm,vlogit,p,visible):
    n,h,w=hm.shape;yy,xx=torch.meshgrid((torch.arange(h,device=hm.device)+.5)/h,(torch.arange(w,device=hm.device)+.5)/w,indexing='ij')
    target=torch.exp(-.5*(((xx[None]-p[:,0,None,None])*w/1.3)**2+((yy[None]-p[:,1,None,None])*h/1.3)**2)).flatten(1)
    target=target/target.sum(1,keepdim=True).clamp_min(1e-8)
    loc=-(target*hm.flatten(1).log_softmax(1)).sum(1)
    coord=F.smooth_l1_loss(coordinates(hm),p,reduction='none').sum(1)
    point=((loc+20*coord)*visible).sum()/visible.sum().clamp_min(1)
    bce=F.binary_cross_entropy_with_logits(vlogit,visible)
    return point+.7*bce,point,bce

def main():
    torch.set_num_threads(4);torch.manual_seed(17);np.random.seed(17);random.seed(17);torch.backends.cudnn.benchmark=True
    device='cuda' if torch.cuda.is_available() else 'cpu';net=SupportNet().to(device)
    train=DataLoader(Samples(TRAIN,True),batch_size=32,shuffle=True,num_workers=0);val=DataLoader(Samples(VAL),batch_size=32,num_workers=0)
    optimizer=torch.optim.AdamW([{'params':net.layer2.parameters(),'lr':.00008},{'params':net.layer3.parameters(),'lr':.00015},{'params':list(net.heat.parameters())+list(net.visible.parameters()),'lr':.0006}],weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,60,eta_min=.00001)
    best=1e9;history=[];start=time.perf_counter();best_epoch=0
    for epoch in range(1,61):
        net.train();tl=[]
        for x,p,v,r in train:
            x,p,v=x.to(device),p.to(device),v.to(device);optimizer.zero_grad(set_to_none=True);hm,vl=net(x);ll,_,_=loss(hm,vl,p,v);ll.backward();nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step();tl.append(ll.item())
        net.eval();losses=[];errs=[];vis_acc=[]
        with torch.inference_mode():
            for x,p,v,r in val:
                x,p,v=x.to(device),p.to(device),v.to(device);hm,vl=net(x);ll,pl,bl=loss(hm,vl,p,v);losses.append([ll.item(),pl.item(),bl.item()]);pp=coordinates(hm);e=((pp-p).cpu()*(r[:,2:]-r[:,:2])).norm(dim=1);errs.extend(e[v.cpu()>0].tolist());vis_acc.extend(((vl.sigmoid()>.5)==(v>.5)).cpu().tolist())
        score=float(np.mean([t[0] for t in losses]));rec={'epoch':epoch,'train_loss':float(np.mean(tl)),'validation_loss':score,'validation_median_pixel_error':float(np.median(errs)),'validation_reportability_accuracy_at_05':float(np.mean(vis_acc)),'seconds':round(time.perf_counter()-start,1)};history.append(rec)
        if score<best:
            best=score;best_epoch=epoch;torch.save({'state_dict':net.state_dict(),'epoch':epoch,'validation_loss':score,'size':SIZE,'label_sha256':hashlib.sha256((OUT/'labels.json').read_bytes()).hexdigest()},OUT/'support_best.pt')
        scheduler.step()
        (OUT/'training_history.json').write_text(json.dumps(history,indent=2))
        if epoch%5==0 or epoch==1:print(json.dumps(rec),flush=True)
        if epoch-best_epoch>=18 and epoch>=30:break
    (OUT/'training.json').write_text(json.dumps({'architecture':'ImageNet ResNet18 through layer3; spatial heatmap + binary reportability head','pretrained':'torchvision ResNet18 IMAGENET1K_V1','train_frames':TRAIN,'validation_frames':VAL,'training_crops':240,'visible_training_points':read(OUT/'label_counts.json')['train']['visible'],'best_epoch':best_epoch,'epochs_run':epoch,'seconds':time.perf_counter()-start,'device':device,'seed':17,'selection':'Lowest validation heatmap+coordinate loss plus 0.7*reportability BCE; no test samples used.','augmentation':'4 crop-jitter/flip/color variants per labeled box each epoch; these are not additional labeled examples.','frozen':'YOLO detector, IDs, stage calibration; early ResNet layers and batchnorm statistics.'},indent=2))
    print('Training finished; selected epoch',best_epoch,flush=True)

if __name__=='__main__':main()
