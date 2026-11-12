"""Small supervised identity experiment, using only existing development training IDs."""
from reid_trial import *
import random,time
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler

class Data(Dataset):
    def __init__(self,split,augment=False):
        d=read(OUT/'data/development_v1.json')['controlled'];self.rows=[];self.images={};self.augment=augment
        for k in d[split]:
            self.images[k]=cv2.imread(str(ROOT/'Dance_Project_Shared/dataset/frames'/f'frame_{4420+k:06}.png'))
            self.rows.extend((k,o) for o in d['labels'][str(k)] if o['box'] is not None and not o.get('outside',False))
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        k,o=self.rows[i];b=np.array(o['box'],float);wh=b[2:]-b[:2]
        if self.augment:b+=np.r_[np.random.uniform(-.05,.05,2)*wh,np.random.uniform(-.05,.05,2)*wh]
        l,t,r,bottom=map(round,b);im=self.images[k][max(0,t):min(720,bottom),max(0,l):min(1280,r)];im=cv2.cvtColor(cv2.resize(im,(128,256)),cv2.COLOR_BGR2RGB).astype(np.float32)/255
        if self.augment:
            if random.random()<.5:im=im[:,::-1].copy()
            im=np.clip(im*random.uniform(.65,1.25)+random.uniform(-.05,.05),0,1)
        im=(im-np.array([.485,.456,.406]))/np.array([.229,.224,.225]);return torch.from_numpy(im.transpose(2,0,1)).float(),o['id']-1

@torch.inference_mode()
def features(net,loader):
    ff=[];yy=[]
    for x,y in loader:ff.append(torch.nn.functional.normalize(net(x.cuda()),dim=1));yy.append(y.cuda())
    return torch.cat(ff),torch.cat(yy)

def train(scope):
    out=OUT/'reid'/f'tuned_{scope}';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():return
    torch.manual_seed(17);np.random.seed(17);random.seed(17);net=model();train=Data('train',True);clean=Data('train');valid=Data('val');sampler=WeightedRandomSampler(torch.ones(len(train)),len(train)*4,replacement=True);loader=DataLoader(train,batch_size=32,sampler=sampler);cl=DataLoader(clean,batch_size=64);vl=DataLoader(valid,batch_size=64)
    for p in net.parameters():p.requires_grad=False
    blocks=[net.fc] if scope=='fc' else [net.conv3,net.conv4,net.conv5,net.fc]
    for block in blocks:
        for p in block.parameters():p.requires_grad=True
    weights=torch.nn.Parameter(torch.randn(20,512,device='cuda')*.01);opt=torch.optim.AdamW([{'params':[p for p in net.parameters() if p.requires_grad],'lr':.00005},{'params':[weights],'lr':.002}],weight_decay=.001);history=[];best=-1;bestepoch=0;start=time.perf_counter()
    for epoch in range(1,61):
        net.eval();losses=[]
        for x,y in loader:
            opt.zero_grad(set_to_none=True);f=torch.nn.functional.normalize(net(x.cuda()),dim=1);logits=16*f@torch.nn.functional.normalize(weights,dim=1).T;loss=torch.nn.functional.cross_entropy(logits,y.cuda(),label_smoothing=.1);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();losses.append(loss.item())
        tf,ty=features(net,cl);vf,vy=features(net,vl);prototypes=torch.stack([tf[ty==i].mean(0) for i in range(20)]);prototypes=torch.nn.functional.normalize(prototypes,dim=1);correct=int(((vf@prototypes.T).argmax(1)==vy).sum());record={'epoch':epoch,'loss':float(np.mean(losses)),'validation_correct_identity':correct,'validation_refs':len(vy),'seconds':time.perf_counter()-start};history.append(record);write(out/'history.json',history)
        if correct>best:
            best=correct;bestepoch=epoch;torch.save({'state_dict':net.state_dict(),'prototypes':prototypes.cpu(),'epoch':epoch,'scope':scope,'validation':record},out/'best.pt')
        if epoch%10==0 or epoch==1:print('ReID fine tune',scope,record,flush=True)
        if epoch-bestepoch>=15 and epoch>=25:break
    write(out/'complete.json',{'train_crops':len(train),'val_crops':len(valid),'best_epoch':bestepoch,'best_correct':best,'epochs_run':epoch,'seconds':time.perf_counter()-start,'scope':'Existing fourteen controlled training frames with consistent IDs. Six development validation frames select checkpoint by20-class training-prototype retrieval. No new labels, no reserved data used. BN statistics frozen; only chosen backbone blocks and a cosine classification head are trained.'})

if __name__=='__main__':
    torch.set_num_threads(4);cv2.setNumThreads(2)
    for scope in ['fc','tail']:train(scope)
