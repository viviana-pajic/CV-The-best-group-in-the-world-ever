"""Appearance-confirmed recovery for lost roster IDs. Development experiment only."""
from tracker_lab import *
from flow_tracker import displacement
from tracking_runtime import run_roster
import torch

def recovery_associate(tracks,det,k,cfg):
    matches,unused=associate(tracks,det,k,cfg)
    gallery=np.stack([t.recovery_prototype for t in tracks])
    proposed={}
    for j in sorted(unused):
        if det[j]['score']<cfg['recovery_detection_conf']:continue
        similarity=gallery@np.asarray(det[j]['feature']);order=np.argsort(similarity)[::-1];i=int(order[0]);t=tracks[i]
        if i in matches or k-t.last<cfg['recovery_min_age']:continue
        if similarity[i]<cfg['recovery_similarity'] or similarity[i]-similarity[order[1]]<cfg['recovery_margin']:continue
        if i not in proposed or similarity[i]>proposed[i][1]:proposed[i]=(j,float(similarity[i]))
    for i,(j,score) in proposed.items():
        t=tracks[i];box=np.asarray(det[j]['box']);consistent=t.recovery_frame==k-1 and (overlap(t.recovery_box,box)>.2 or np.linalg.norm(core.center(t.recovery_box)-core.center(box))<30)
        t.recovery_streak=t.recovery_streak+1 if consistent else 1;t.recovery_box=box;t.recovery_frame=k
        if t.recovery_streak>=cfg['recovery_confirmation']:
            matches[i]=j;unused.remove(j);t.recovery_events.append({'frame':k,'score':score,'age_before':k-t.last,'box':box.tolist()});t.recovery_streak=0
    return matches,unused

def run_recovery(pred,seeds,features,cfg,flow,prototypes):
    prepared={k:[dict(o,feature=f) for o,f in zip(row,features['frames'][k])] for k,row in pred.items()};sf={o['id']:np.asarray(f) for o,f in zip(seeds,features['seeds'])};created=[]
    class Dancer(core.Dancer):
        def __init__(self,o):
            super().__init__(o);self.appearance=sf[o['id']].copy();self.initial_appearance=self.appearance.copy();self.flow_box=self.box.copy();self.flow_k=0;self.recovery_prototype=prototypes[o['id']-1];self.recovery_frame=-99;self.recovery_streak=0;self.recovery_box=self.box.copy();self.recovery_events=[];created.append(self)
        def predicted(self,k):
            pb,pg=super().predicted(k);age=k-self.last
            if k>self.flow_k:self.flow_box+=np.tile(displacement(flow[k],self.flow_box),2);self.flow_k=k
            if age<=cfg['flow_max_gap']:pb=(1-cfg['flow_blend'])*pb+cfg['flow_blend']*self.flow_box;pg=stage(bottom(pb))
            return pb,pg
        def update(self,b,k):super().update(b,k);self.flow_box=b.copy();self.flow_k=k
    result=run_roster(prepared,seeds,cfg,make_track=Dancer,association=recovery_associate)
    result['recovery_events']=[dict(event,id=t.id) for t in created for event in t.recovery_events];return result

def main():
    out=OUT/'recovery';out.mkdir(exist_ok=True);folder=OUT/'detector/sequences/posture_frozen_960';pred=read(folder/'merged.json');labels=read(ROOT/'Dance_Corrected_Pilot/labels.json');seeds=labels['0'];features=read(OUT/'reid/tuned_tail/development_features.json');checkpoint=torch.load(OUT/'reid/tuned_tail/best.pt',map_location='cpu',weights_only=True);prototypes=checkpoint['prototypes'].numpy();flow=np.load(OUT/'tracking/dense_flow.npz')['flow'].astype(np.float32);base=read(OUT/'reid/tuned_tracking_results.json')['chosen'];results=[]
    for sim,margin,age in itertools.product([.7,.8,.9],[.02,.05],[10,25]):
        cfg={**base['config'],'recovery_similarity':sim,'recovery_margin':margin,'recovery_min_age':age,'recovery_confirmation':3,'recovery_detection_conf':.2};name=f'recovery_s{sim}_m{margin}_age{age}';tracks=run_recovery(pred,seeds,features,cfg,flow,prototypes);score=core.evaluate(tracks,labels,[165,175,185,195,285,299]);r={'name':name,'config':cfg,'validation':score,'recovery_events':tracks['recovery_events']};results.append(r);write(out/f'{name}_tracks.json',tracks);write(out/'development_partial.json',results);print(name,score['correct_initial_ids'],score['wrong_initial_ids'],len(tracks['recovery_events']),flush=True)
    chosen=max(results,key=lambda r:(r['validation']['correct_initial_ids'],-r['validation']['wrong_initial_ids'],-len(r['recovery_events']),r['config']['recovery_similarity']))
    write(out/'development_results.json',{'scope':'Development only. Recovery uses training-only identity prototypes and requires three consecutive appearance matches with a margin over all other known IDs. No later manual identity or support coordinates feed tracking.','baseline':base,'candidates':results,'chosen':chosen});print('RECOVERY CHOSEN',chosen['name'],chosen['validation'],flush=True)

if __name__=='__main__':main()
