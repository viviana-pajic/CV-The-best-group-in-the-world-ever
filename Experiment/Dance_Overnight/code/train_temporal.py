"""Matched current-frame control versus +/-2-frame support context, development only."""
from train_support import *
from conditioned_support import TemporalSupportNet
from collections import Counter

class TemporalData(Data):
    def __init__(self,manifest,split,augment,radius):
        super().__init__(manifest,split,augment,wide=True,ignore_uncertain=True);self.radius=radius;self.context={};cache={}
        for clip,d in manifest.items():
            for k in d[split]:
                source=d['source_first']+k;frames=[]
                for f in [source-radius,source,source+radius]:
                    if f not in cache:cache[f]=cv2.imread(str(ROOT/'Dance_Project_Shared/dataset/frames'/f'frame_{f:06}.png'));assert cache[f] is not None
                    frames.append(cache[f])
                self.context[(clip,k)]=frames
        self.images={}
    def __getitem__(self,i):
        key,o=self.rows[i];r=roi(o['box'],self.augment)
        if self.augment:
            size=r[2:]-r[:2];r+=np.tile(np.random.uniform(-.07,.07,2)*size,2);r[:2]=np.maximum(r[:2],0);r[2:]=np.minimum(r[2:],[1280,720])
        vis=float(o['floor'] is not None);p=(np.array(o['floor'],np.float32)-r[:2])/(r[2:]-r[:2]) if vis else np.array([.5,.5],np.float32)
        if vis and (p.min()<=.01 or p.max()>=.99):r=roi(o['box']);p=(np.array(o['floor'],np.float32)-r[:2])/(r[2:]-r[:2])
        patches=[crop(im,r) for im in self.context[key]]
        if self.augment:
            if random.random()<.5:patches=[im[:,::-1].copy() for im in patches];p[0]=1-p[0]
            gain=random.uniform(.7,1.25);bias=random.uniform(-15,15);patches=[np.clip(im.astype(np.float32)*gain+bias,0,255).astype(np.uint8) for im in patches]
            if random.random()<.15:
                sigma=random.uniform(.3,1.0);patches=[cv2.GaussianBlur(im,(3,3),sigma) for im in patches]
        assert not vis or (p.min()>0 and p.max()<1)
        return torch.cat([tensor(im) for im in patches],0),torch.tensor(p,dtype=torch.float32),torch.tensor(vis),torch.tensor(r),torch.tensor(o['support_type']=='knees')

def train_temporal(name,radius):
    folder=OUT/'runs'/name;folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():return
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(17);np.random.seed(17);random.seed(17);torch.backends.cudnn.benchmark=True;manifest_path=OUT/'data/development_ending.json';manifest=read(manifest_path);ds=TemporalData(manifest,'train',True,radius);vs=TemporalData(manifest,'val',False,radius);groups=Counter(ds.groups);sampler=WeightedRandomSampler([1/max(groups[g],30) for g in ds.groups],len(ds)*4,replacement=True);dl=DataLoader(ds,batch_size=32,sampler=sampler,num_workers=0);vl=DataLoader(vs,batch_size=32,num_workers=0);selected=read(OUT/'ending_experiment/results.json')['chosen']['candidate'];source=OUT/'runs'/selected['run']/selected['checkpoint'];net=TemporalSupportNet(torch.load(source,map_location='cpu',weights_only=True)['state_dict']).cuda();opt=torch.optim.AdamW([{'params':net.stem[0].parameters(),'lr':.00001},{'params':net.layer2.parameters(),'lr':.00002},{'params':net.layer3.parameters(),'lr':.00004},{'params':list(net.heat.parameters())+list(net.visible.parameters()),'lr':.00015}],weight_decay=.001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,30,eta_min=.000005)
    settings={'name':name,'temporal_input':True,'temporal_radius':radius,'contact_attention':True,'target_box_mask':False,'initialization':str(source.relative_to(OUT)),'initialization_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'manifest':str(manifest_path),'data_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'training_boxes':len(ds),'validation_boxes':len(vs),'seed':17,'max_epochs':30,'scope':'Matched9-channel architecture. Center RGB conv initialized from V6; neighboring conv weights zero. Radius0 repeats current frame, radius2 adds +/-80ms context. Labels and splits unchanged. Whole-video operation is offline. No confirmation labels used.'};write(folder/'settings.json',settings);history=[];best=1e9;joint=-1;bestepoch=0;start=time.perf_counter()
    settings['scope']=f"Matched9-channel architecture. Center RGB conv initialized from V6; neighboring conv weights zero. Radius0 repeats current frame; this run uses radius{radius} ({radius*40}ms each side). Labels and splits unchanged. Whole-video operation is offline. No confirmation labels used."
    write(folder/'settings.json',settings)
    for epoch in range(1,31):
        net.train();tl=[]
        for x,p,v,r,posture in dl:
            x,p,v=x.cuda(),p.cuda(),v.cuda();opt.zero_grad(set_to_none=True);hm,visibility=net(x);_,pl,bl=loss(hm,visibility,p,v);ll=pl+.7*bl;ll.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();tl.append(ll.item())
        net.eval();val=validate(net,vl,.7);history.append({'epoch':epoch,'train_loss':float(np.mean(tl)),'validation':val,'seconds':time.perf_counter()-start});checkpoint={'state_dict':net.state_dict(),'epoch':epoch,'settings':settings,'validation':val}
        if val['loss']<best:best=val['loss'];bestepoch=epoch;torch.save(checkpoint,folder/'best.pt')
        if val['selected']['joint_f1']>joint:joint=val['selected']['joint_f1'];torch.save(checkpoint,folder/'best_joint.pt')
        sched.step();write(folder/'history.json',history)
        if epoch==1 or epoch%5==0:print(name,epoch,'joint',round(val['selected']['joint_f1'],4),'seconds',round(time.perf_counter()-start),flush=True)
        if epoch>=20 and epoch-bestepoch>=12:break
    write(folder/'complete.json',{'best_epoch':bestepoch,'epochs_run':epoch,'seconds':time.perf_counter()-start,'best_joint_f1':joint});print('FINISHED temporal',name,flush=True)

if __name__=='__main__':
    while not (OUT/'efficiency/results.json').exists() or len(read(OUT/'efficiency/results.json')['candidates'])<3:time.sleep(5)
    while not (OUT/'late_full_replay/provenance.json').exists():time.sleep(5)
    train_temporal('v8_repeated_current_s17',0);train_temporal('v8_temporal2_s17',2)
