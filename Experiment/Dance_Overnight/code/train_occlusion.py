"""Synthetic-contact-occlusion ablation. Hidden real contacts never receive invented targets."""
from train_support import *
from torch.nn import functional as F
from collections import Counter

class OccludedData(Data):
    def __init__(self,*args,probability=.35,deterministic=False,**kwargs):
        super().__init__(*args,**kwargs);self.probability=probability;self.deterministic=deterministic
    def __getitem__(self,i):
        x,p,visible,r,posture=super().__getitem__(i);point_valid=visible.clone();rng=np.random.RandomState(9173+i) if self.deterministic else np.random;synthetic=False
        if visible.item() and rng.random()<self.probability:
            # Hide a horizontal strip around the known support. This is a controlled
            # stress augmentation, not a photorealistic model of another dancer.
            h,w=x.shape[-2:];top=max(0,int((p[1].item()-rng.uniform(.10,.24))*h));end=min(h,int((p[1].item()+rng.uniform(.06,.16))*h)+1)
            fill=torch.tensor(rng.uniform(-1.5,.5,(3,1,1)),dtype=x.dtype);noise=torch.tensor(rng.normal(0,.025,(3,end-top,w)),dtype=x.dtype);x=x.clone();x[:3,top:end]=fill+noise;visible=torch.tensor(0.);synthetic=True
        return x,p,visible,r,posture,point_valid,torch.tensor(synthetic)

@torch.inference_mode()
def masked_validate(net,loader):
    errors=[];reportability=[]
    for x,p,v,r,posture,pvalid,synthetic in loader:
        hm,vl=net(x.cuda());pred=coordinates(hm).cpu().numpy();rr=r.numpy();points=p.numpy();q=rr[:,:2]+pred*(rr[:,2:]-rr[:,:2]);target=rr[:,:2]+points*(rr[:,2:]-rr[:,:2]);scores=vl.sigmoid().cpu().numpy()
        for i in np.flatnonzero(synthetic.numpy()):errors.append(float(np.linalg.norm(stage(q[i])-stage(target[i]))));reportability.append(float(scores[i]))
    return {'synthetic_references':len(errors),'correct1':sum(e<=1 for e in errors),'correct3':sum(e<=3 for e in errors),'median_error_pct':float(np.median(errors)),'p90_error_pct':float(np.percentile(errors,90)),'reported_visible_at_05':sum(v>=.5 for v in reportability),'scope':'Artificial horizontal masks on visible, labeled validation contacts. Point targets are known before masking. This does not measure real hidden-contact accuracy.'}

def train(probability):
    name=f'v7_occlusion_p{probability}_s17';out=OUT/'runs'/name
    if (out/'complete.json').exists():return
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(17);np.random.seed(17);random.seed(17);torch.backends.cudnn.benchmark=True
    manifest=read(OUT/'data/development_ending.json');ds=OccludedData(manifest,'train',augment=True,ignore_uncertain=True,probability=probability);vs=Data(manifest,'val',ignore_uncertain=True);masked=OccludedData(manifest,'val',ignore_uncertain=True,probability=1.,deterministic=True);counts=Counter(ds.groups);weights=[1/max(counts[g],30) for g in ds.groups];loader=DataLoader(ds,batch_size=32,sampler=WeightedRandomSampler(weights,len(ds)*4,replacement=True),num_workers=0);vl=DataLoader(vs,batch_size=32);ml=DataLoader(masked,batch_size=32)
    net=AttentionSupportNet(condition=False).cuda();groups=[{'params':net.layer2.parameters(),'lr':.00008},{'params':net.layer3.parameters(),'lr':.00015},{'params':list(net.heat.parameters())+list(net.visible.parameters()),'lr':.0006}];opt=torch.optim.AdamW(groups,weight_decay=.001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,60,eta_min=.00001);settings={'name':name,'target_box_mask':False,'contact_attention':True,'seed':17,'synthetic_contact_occlusion_probability':probability,'training_boxes':len(ds),'validation_boxes':len(vs),'manifest':'data/development_ending.json','selection':'Clean development joint F1; separate masked-point checkpoint only while clean F1 >= .90.','point_supervision':'Visible originals retain a known coordinate when synthetically hidden; visibility becomes zero. Real hidden and airborne examples have no coordinate loss. Uncertain rows excluded.','reference':'https://arxiv.org/abs/1708.04552','limitation':'Cutout-inspired augmentation; horizontal masks are not realistic overlapping dancers.'};write(out/'settings.json',settings);history=[];best_loss=1e9;best_epoch=0;best_joint=-1;best_masked=-1;start=time.perf_counter()
    for epoch in range(1,61):
        net.train();losses=[]
        for x,p,v,r,posture,pvalid,synthetic in loader:
            x,p,v,pvalid=x.cuda(),p.cuda(),v.cuda(),pvalid.cuda();opt.zero_grad(set_to_none=True);hm,logit=net(x);_,point_loss,_=loss(hm,logit,p,pvalid);ll=point_loss+.7*F.binary_cross_entropy_with_logits(logit,v);ll.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();losses.append(ll.item())
        net.eval();clean=validate(net,vl,.7);hidden=masked_validate(net,ml);record={'epoch':epoch,'train_loss':float(np.mean(losses)),'validation':clean,'synthetic_occlusion':hidden,'seconds':time.perf_counter()-start};history.append(record);ck={'state_dict':net.state_dict(),'epoch':epoch,'validation':clean,'synthetic_occlusion':hidden,'settings':settings}
        if clean['loss']<best_loss:best_loss=clean['loss'];best_epoch=epoch;torch.save(ck,out/'best.pt')
        if clean['selected']['joint_f1']>best_joint:best_joint=clean['selected']['joint_f1'];torch.save(ck,out/'best_joint.pt')
        if clean['selected']['joint_f1']>=.90 and hidden['correct3']>best_masked:best_masked=hidden['correct3'];torch.save(ck,out/'best_masked.pt')
        scheduler.step();write(out/'history.json',history)
        if epoch==1 or epoch%10==0:print(name,epoch,'clean',round(clean['selected']['joint_f1'],3),'masked',hidden['correct3'],hidden['synthetic_references'],'seconds',round(record['seconds']),flush=True)
        if epoch-best_epoch>=22 and epoch>=40:break
    write(out/'complete.json',{'epochs_run':epoch,'best_epoch':best_epoch,'best_joint_f1':best_joint,'best_masked_correct3':best_masked,'seconds':time.perf_counter()-start});print('FINISHED',name,flush=True)

if __name__=='__main__':
    # Keep one GPU job at a time; earlier development evaluation must finish first.
    while not (OUT/'ending_experiment/results.json').exists():time.sleep(5)
    for probability in [.3,.6]:train(probability)
