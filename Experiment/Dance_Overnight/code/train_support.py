"""Reproducible overnight support-model experiments, development data only."""
import sys,os,json,random,time,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Dance_Overnight'
sys.path.insert(0,str(ROOT/'Dance_Floor_Contact'))
import cv2,numpy as np,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from support_train import SupportNet,roi,crop,tensor,coordinates,loss
from compare import stage
from conditioned_support import ConditionedSupportNet,AttentionSupportNet,target_mask

def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2),encoding='utf-8')

class Data(Dataset):
    def __init__(self,manifest,split,augment=False,wide=False,condition=False,ignore_uncertain=False):
        self.rows=[];self.images={};self.augment=augment;self.wide=wide;self.condition=condition
        for clip,d in manifest.items():
            for k in d[split]:
                frame=d['source_first']+k;key=(clip,k)
                self.images[key]=cv2.imread(str(ROOT/'Dance_Project_Shared/dataset/frames'/f'frame_{frame:06}.png'))
                assert self.images[key] is not None,frame
                self.rows.extend((key,o) for o in d['labels'][str(k)] if o['box'] is not None and not (ignore_uncertain and o['contact']=='uncertain'))
        self.groups=[o['support_type'] if o['floor'] is not None else 'hidden' for _,o in self.rows]
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        key,o=self.rows[i];r=roi(o['box'],self.augment)
        if self.augment and self.wide:
            size=r[2:]-r[:2];r+=np.tile(np.random.uniform(-.07,.07,2)*size,2)
            r[:2]=np.maximum(r[:2],0);r[2:]=np.minimum(r[2:],[1280,720])
        vis=float(o['floor'] is not None);p=(np.array(o['floor'],np.float32)-r[:2])/(r[2:]-r[:2]) if vis else np.array([.5,.5],np.float32)
        if vis and (p.min()<=.01 or p.max()>=.99):r=roi(o['box']);p=(np.array(o['floor'],np.float32)-r[:2])/(r[2:]-r[:2])
        im=crop(self.images[key],r);flip=False
        if self.augment:
            if random.random()<.5:im=im[:,::-1].copy();p[0]=1-p[0];flip=True
            im=np.clip(im.astype(np.float32)*random.uniform(.7,1.25)+random.uniform(-15,15),0,255).astype(np.uint8)
            if self.wide and random.random()<.15:im=cv2.GaussianBlur(im,(3,3),random.uniform(.3,1.0))
        assert not vis or (p.min()>0 and p.max()<1),(key,o['id'],p)
        x=tensor(im)
        if self.condition:x=torch.cat([x,target_mask(o['box'],r,flip)],0)
        return x,torch.tensor(p,dtype=torch.float32),torch.tensor(vis),torch.tensor(r),torch.tensor(1 if o['support_type']=='knees' else 0)

@torch.inference_mode()
def validate(net,loader,visibility_weight):
    all_loss=[];error=[];vis_p=[];vis_gt=[];postures=[]
    for x,p,v,r,posture in loader:
        x,p,v=x.cuda(),p.cuda(),v.cuda();hm,vl=net(x);_,pl,bl=loss(hm,vl,p,v);all_loss.append((pl+visibility_weight*bl).item());pp=coordinates(hm).cpu().numpy();r=r.numpy();target=p.cpu().numpy();v=v.cpu().numpy()
        global_pred=r[:,:2]+pp*(r[:,2:]-r[:,:2]);global_ref=r[:,:2]+target*(r[:,2:]-r[:,:2]);error.extend(float(np.linalg.norm(stage(a)-stage(b))) if vv else None for a,b,vv in zip(global_pred,global_ref,v));vis_p.extend(vl.sigmoid().cpu().tolist());vis_gt.extend(v.tolist());postures.extend(posture.tolist())
    threshold_scores=[]
    for t in [.1,.25,.4,.5,.6,.75,.9]:
        em=np.array(vis_p)>=t;gt=np.array(vis_gt)>0;good=np.array([e is not None and e<=3 for e in error])&em
        tp=int((em&gt).sum());fp=int((em&~gt).sum());fn=int((~em&gt).sum())
        threshold_scores.append({'threshold':t,'correct':int(good.sum()),'refs':int(gt.sum()),'emitted':int(em.sum()),'joint_f1':2*int(good.sum())/max(int(gt.sum()+em.sum()),1),'visibility_f1':2*tp/max(2*tp+fp+fn,1),'hidden_emitted':fp})
    best=max(threshold_scores,key=lambda x:(x['joint_f1'],x['visibility_f1']))
    return {'loss':float(np.mean(all_loss)),'raw_median_pct_width':float(np.median([e for e in error if e is not None])),'threshold_options':threshold_scores,'selected':best}

def train(name,manifest_path,seed=17,balance=True,wide=False,vis_weight=.7,epochs=80,condition=False,attention=False,ignore_uncertain=False,weight_floor=0):
    torch.set_num_threads(4);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);torch.backends.cudnn.benchmark=True
    out=OUT/'runs'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():print('Already complete:',name,flush=True);return
    manifest=read(manifest_path);ds=Data(manifest,'train',True,wide,condition,ignore_uncertain);vs=Data(manifest,'val',condition=condition,ignore_uncertain=ignore_uncertain)
    from collections import Counter
    groupcounts=Counter(ds.groups);weights=[1/max(groupcounts[g],weight_floor) if balance else 1 for g in ds.groups]
    sampler=WeightedRandomSampler(weights,len(ds)*4,replacement=True)
    trainloader=DataLoader(ds,batch_size=32,sampler=sampler,num_workers=0);valloader=DataLoader(vs,batch_size=32,num_workers=0)
    net=(AttentionSupportNet(condition=condition) if attention else ConditionedSupportNet() if condition else SupportNet()).cuda();groups=[{'params':net.layer2.parameters(),'lr':.00008},{'params':net.layer3.parameters(),'lr':.00015},{'params':list(net.heat.parameters())+list(net.visible.parameters()),'lr':.0006}]
    if condition:groups.append({'params':net.stem[0].parameters(),'lr':.00001})
    opt=torch.optim.AdamW(groups,weight_decay=.001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=.00001)
    settings={'name':name,'manifest':str(manifest_path),'data_sha256':hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),'seed':seed,'balance_posture':balance,'wide_augmentation':wide,'visibility_loss_weight':vis_weight,'max_epochs':epochs,'training_boxes':len(ds),'validation_boxes':len(vs),'training_groups':dict(groupcounts),'selection':'Lowest validation combined loss. Threshold maximizes visible-point joint F1 on development validation only.','initialization':'ImageNet ResNet18, no choreography-specific checkpoint'};write(out/'settings.json',settings)
    settings['target_box_mask']=condition;settings['contact_attention']=attention;settings['ignore_uncertain']=ignore_uncertain;settings['sampling_count_floor']=weight_floor;write(out/'settings.json',settings)
    best=1e9;bestepoch=0;bestjoint=-1;history=[];start=time.perf_counter()
    for epoch in range(1,epochs+1):
        net.train();tl=[]
        for x,p,v,r,posture in trainloader:
            x,p,v=x.cuda(),p.cuda(),v.cuda();opt.zero_grad(set_to_none=True);hm,vl=net(x);_,pl,bl=loss(hm,vl,p,v);ll=pl+vis_weight*bl;ll.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();tl.append(ll.item())
        net.eval();v=validate(net,valloader,vis_weight);rec={'epoch':epoch,'train_loss':float(np.mean(tl)),'validation':v,'seconds':round(time.perf_counter()-start,1)};history.append(rec)
        if v['loss']<best:
            best=v['loss'];bestepoch=epoch;torch.save({'state_dict':net.state_dict(),'epoch':epoch,'validation':v,'settings':settings},out/'best.pt');write(out/'best_validation.json',v)
        if v['selected']['joint_f1']>bestjoint:
            bestjoint=v['selected']['joint_f1'];torch.save({'state_dict':net.state_dict(),'epoch':epoch,'validation':v,'settings':settings},out/'best_joint.pt');write(out/'best_joint_validation.json',v)
        sched.step();write(out/'history.json',history)
        if epoch==1 or epoch%10==0:print(name,epoch,'loss',round(v['loss'],3),'jointF1',round(v['selected']['joint_f1'],3),'elapsed',rec['seconds'],flush=True)
        if epoch-bestepoch>=22 and epoch>=40:break
    write(out/'complete.json',{'best_epoch':bestepoch,'epochs_run':epoch,'seconds':time.perf_counter()-start,'validation':read(out/'best_validation.json')});print('FINISHED',name,bestepoch,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--suite',default='v1');args=parser.parse_args();m=OUT/'data/development_v1.json'
    if args.suite=='v1':
        for name,seed,balance,wide,vw in [('balanced_s17',17,True,False,.7),('balanced_s29',29,True,False,.7),('wide_s17',17,True,True,.7),('visibility_s17',17,True,True,1.5),('unbalanced_control',17,False,False,.7)]:train('v1_'+name,m,seed,balance,wide,vw)
    elif args.suite=='v2':
        m=OUT/'data/development_v2.json'
        for name,seed,balance,wide,vw in [('balanced_s17',17,True,False,.7),('balanced_s29',29,True,False,.7),('wide_s17',17,True,True,.7),('visibility_s17',17,True,True,1.5),('unbalanced_control',17,False,False,.7)]:train('v2_'+name,m,seed,balance,wide,vw)
    elif args.suite=='v3':
        m=OUT/'data/development_v2.json'
        for name,seed,balance,wide,vw in [('mask_s17',17,True,False,.7),('mask_s29',29,True,False,.7),('mask_wide_s17',17,True,True,.7)]:train('v3_'+name,m,seed,balance,wide,vw,condition=True)
    elif args.suite=='v4':
        m=OUT/'data/development_v2.json'
        for name,seed,condition in [('attention_s17',17,False),('mask_attention_s17',17,True),('mask_attention_s29',29,True)]:train('v4_'+name,m,seed,condition=condition,attention=True)
    elif args.suite=='v5':
        m=OUT/'data/development_v2.json'
        train('v5_clean_v2_s17',m,ignore_uncertain=True)
        m=OUT/'data/development_expanded.json'
        for name,seed,condition,attention,wide in [('clean_s17',17,False,False,False),('clean_s29',29,False,False,False),('mask_s17',17,True,False,False),('attention_s17',17,False,True,False),('mask_attention_s17',17,True,True,False),('wide_s17',17,False,False,True)]:
            train('v5_'+name,m,seed,condition=condition,attention=attention,wide=wide,ignore_uncertain=True,weight_floor=30)
