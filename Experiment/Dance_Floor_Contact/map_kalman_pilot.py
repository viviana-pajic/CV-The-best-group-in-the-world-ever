"""Causal map filtering only. Frozen boxes, IDs and association are never modified."""
import copy,json,hashlib,time,subprocess
from pathlib import Path
import numpy as np
import cv2
from compare import stage,bottom,match_boxes,overlap
from corrected_setup import OUT as LABEL_DIR,SOURCE,VAL,TEST,read
from occlusion_pilot import OUT as TRACK_DIR,inside,evaluate
from occlusion_deliver import source_frame,draw_map
from render_controlled import text,BG,INK,MUTED

OUT=TRACK_DIR.parent/'Dance_Map_Kalman_Pilot';OUT.mkdir(exist_ok=True)
FPS=25;DT=1/FPS

def write(name,obj):
    def enc(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x))
    (OUT/name).write_text(json.dumps(obj,default=enc,indent=2),encoding='utf-8')

def limit(v,n):return v*min(1,n/max(np.linalg.norm(v),1e-9))

class MapFilter:
    def __init__(self,cfg):
        self.cfg=cfg;self.x=None;self.P=None;self.last=-1000;self.pending=[]
        self.measurements=[];self.last_measurement=-1000
        self.F=np.eye(4);self.F[0,2]=self.F[1,3]=DT
        self.H=np.eye(4)[:2]
        g=np.array([[DT*DT/2,0],[0,DT*DT/2],[DT,0],[0,DT]])
        self.Q=g@g.T*cfg['acceleration_sigma']**2+np.eye(4)*1e-6
        self.R=np.eye(2)*cfg['measurement_sigma']**2

    def step(self,k,z):
        event='no_measurement';accepted=False;nis=None
        if z is not None:
            if k-self.last_measurement>3:self.measurements=[]
            self.measurements.append(z.copy());self.measurements=self.measurements[-self.cfg.get('measurement_median',1):]
            z=np.median(self.measurements,axis=0);self.last_measurement=k
        if self.x is None or k-self.last>25:
            if z is None:return None,'unknown',event,None,None
            self.x=np.r_[z,[0.,0.]];self.P=np.diag([.36,.36,100.,100.]);self.last=k
            self.pending=[]
            return self.x[:2].copy(),'proxy','initialize',0.,.6
        prior=self.x[:2].copy()
        self.x=self.F@self.x;self.P=self.F@self.P@self.F.T+self.Q
        if z is not None:
            residual=z-self.x[:2];S=self.H@self.P@self.H.T+self.R
            nis=float(residual@np.linalg.solve(S,residual))
            accepted=nis<=9.21
            if not accepted:
                if self.pending and (k-self.pending[-1][0]!=1 or np.linalg.norm(z-self.pending[-1][1])>1.5):self.pending=[]
                self.pending.append((k,z.copy()));self.pending=self.pending[-3:]
                if len(self.pending)==3:
                    # A persistent new location may be genuine motion. Increase uncertainty,
                    # then correct with a bounded step, rather than freezing indefinitely.
                    self.P[:2,:2]+=np.eye(2)*1.5
                    accepted=True;event='persistent_change'
                else:event='jump_rejected'
            else:self.pending=[];event='measurement_update'
            if accepted:
                S=self.H@self.P@self.H.T+self.R;K=np.linalg.solve(S,(self.P@self.H.T).T).T
                self.x+=K@residual
                I=np.eye(4)-K@self.H
                self.P=I@self.P@I.T+K@self.R@K.T
                self.x[2:]=limit(self.x[2:],35.)
                # Rendering limit in relative mat units, not an assertion about physical speed.
                delta=self.x[:2]-prior
                if np.linalg.norm(delta)>.875:
                    self.x[:2]=prior+limit(delta,.875);self.P[:2,:2]+=np.eye(2)*.25
                    event+=';correction_limited'
                self.last=k
        else:self.pending=[]
        if not accepted:self.x[2:]*=.85
        # Apply the SAME maximum step as the current map (.35 * 2.5 = .875),
        # including predictions. This prevents comparing two different speed caps.
        self.x[:2]=prior+limit(self.x[:2]-prior,.875)
        age=k-self.last
        if age>25 or not inside(self.x[:2]):return None,'unknown',event,nis,None
        uncertainty=float(np.sqrt(np.linalg.eigvalsh(self.P[:2,:2]).max()))
        return self.x[:2].copy(),'proxy' if accepted else 'estimated',event,nis,uncertainty

def filter_maps(base,cfg):
    result=copy.deepcopy(base);filters={i:MapFilter(cfg) for i in range(1,21)}
    for k,row in enumerate(result['frames']):
        for o in row:
            # Recompute the unsmoothed point from the EXACT frozen box. The old
            # display smoothing/jump gate is replaced, never fed back to the tracker.
            p=stage(bottom(np.array(o['box'])))
            eligible=o['status']=='observed' and inside(p) and not o['possible_occluders']
            f=filters[o['id']];value,state,event,nis,u=f.step(k,p if eligible else None)
            for key in ['raw_map','uncertainty']:o.pop(key,None)
            o.update(map=value,map_status=state,map_age_frames=k-f.last,filter_event=event,innovation_squared=nis)
            if eligible:o['raw_map']=p
            if u is not None:o['uncertainty']=min(u,8)
    result['map_filter']=cfg
    return result

def frozen_hash(data):
    values=[[(o['id'],o['box'],o['status'],o['age_frames'],o['score'],o['possible_occluders']) for o in row] for row in data['frames']]
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()

def motion_stats(a,b,start=1,end=300):
    per_method={};paired=[];solid_paired=[]
    for k in range(max(1,start),end):
        for i in range(20):
            aa=[a['frames'][f][i] for f in [k-1,k]];bb=[b['frames'][f][i] for f in [k-1,k]]
            if all(o.get('map') is not None for o in aa+bb):
                pair=(k,i,float(np.linalg.norm(np.array(aa[1]['map'])-aa[0]['map'])),float(np.linalg.norm(np.array(bb[1]['map'])-bb[0]['map'])))
                paired.append(pair)
                if all(o['map_status']=='proxy' for o in aa+bb):solid_paired.append(pair)
    for name,col in [('current',2),('kalman',3)]:
        jumps=[r[col] for r in paired];solid=[r[col] for r in solid_paired]
        acc=[]
        data=a if name=='current' else b
        for k in range(max(2,start),end):
            for i in range(20):
                shared=[d['frames'][f][i] for d in (a,b) for f in [k-2,k-1,k]]
                if all(o.get('map') is not None for o in shared):
                    pp=[np.array(data['frames'][f][i]['map']) for f in [k-2,k-1,k]]
                    acc.append(float(np.linalg.norm(pp[2]-2*pp[1]+pp[0])))
        per_method[name]={'paired_transitions':len(jumps),'steps_over_half_pct_mat_width':sum(v>.5 for v in jumps),'steps_over_1pct_mat_width':sum(v>1 for v in jumps),'steps_over_2pct_mat_width':sum(v>2 for v in jumps),'p99_step_pct_mat_width':float(np.percentile(jumps,99)),'p99_change_in_step_pct_width':float(np.percentile(acc,99)),'abrupt_step_changes_over_half_pct':sum(v>.5 for v in acc),
            'paired_solid_transitions':len(solid),'solid_steps_over_1pct':sum(v>1 for v in solid),'solid_p99_step_pct':float(np.percentile(solid,99)),
            'proxy_frames':sum(o['map_status']=='proxy' for row in (a if name=='current' else b)['frames'][start:end] for o in row),
            'predicted_frames':sum(o['map_status']=='estimated' for row in (a if name=='current' else b)['frames'][start:end] for o in row)}
    return {'scope':'Same dancer IDs and frame pairs where both maps exist. Large steps are a visual continuity metric, not automatically errors. Solid-only comparison excludes predicted/held positions.','metrics':per_method,'largest_current_steps':sorted(paired,key=lambda r:r[2],reverse=True)[:12]}

def stationary(a,b,labels):
    residuals={'current':{},'kalman':{}};g0={o['id']:o for o in labels['285']};g1={o['id']:o for o in labels['299']}
    for k in range(285,300):
        f=(k-285)/14
        gt=[dict(id=i,box=np.array(o['box'])*(1-f)+np.array(g1[i]['box'])*f,floor=np.array(o['floor'])*(1-f)+np.array(g1[i]['floor'])*f) for i,o in g0.items() if o['floor'] is not None and g1[i]['floor'] is not None]
        for mode,data in [('current',a),('kalman',b)]:
            rows=[o for o in data['frames'][k] if o['status']=='observed' and o['map_status']=='proxy' and o['map'] is not None]
            links=match_boxes([o['box'] for o in gt],[o['box'] for o in rows],.5)
            for i,j in links.items():residuals[mode].setdefault(gt[i]['id'],{})[k]=np.array(rows[j]['map'])-stage(gt[i]['floor'])
    pairs={i:sorted(set(seq)&set(residuals['kalman'].get(i,{}))) for i,seq in residuals['current'].items()}
    pairs={i:kk for i,kk in pairs.items() if len(kk)>=12}
    result={'scope':'Final 0.56 seconds only; spatially matched to interpolated reviewed endpoints. Shared solid outputs only. Fluctuation about mean error, not absolute accuracy.','paired_ids':list(pairs)}
    for mode in residuals:
        vals={}
        for i,kk in pairs.items():
            arr=np.array([residuals[mode][i][k] for k in kk]);vals[i]=float(np.sqrt(np.mean(np.sum((arr-arr.mean(axis=0))**2,axis=1))))
        result[mode]={'per_id':vals,'median_rms_pct_width':float(np.median(list(vals.values()))) if vals else None}
    return result

def delay_check(a,b,labels):
    # Motion timing against existing sparse reviewed points. Interpolation is ONLY
    # an evaluation approximation and never reaches the filter.
    segments=[]
    for ident in range(1,21):
        seq=[]
        def finish(seq):
            if len(seq)>=6 and seq[-1][0]-seq[0][0]>=40:segments.append((ident,seq.copy()))
        for k in sorted(map(int,labels)):
            o=next(o for o in labels[str(k)] if o['id']==ident);row=a['frames'][k][ident-1]
            valid=o['floor'] is not None and not o.get('outside') and row['status']=='observed' and overlap(np.array(o['box']),np.array(row['box']))>=.5
            if not valid or (seq and k-seq[-1][0]>10):finish(seq);seq=[]
            if valid:seq.append((k,stage(o['floor'])))
        finish(seq)
    output=[]
    for ident,seq in segments:
        times=np.array([v[0] for v in seq]);points=np.array([v[1] for v in seq]);span=float(np.linalg.norm(np.ptp(points,axis=0)))
        if span<2:continue
        kk=[k for k in range(times[0]+8,times[-1]-7) if all(d['frames'][k+s][ident-1].get('map') is not None and d['frames'][k+s][ident-1]['status']=='observed' for d in (a,b) for s in range(-8,9))]
        if len(kk)<20:continue
        ref=np.c_[np.interp(kk,times,points[:,0]),np.interp(kk,times,points[:,1])]
        row={'id':ident,'start':int(times[0]),'end':int(times[-1]),'samples':len(kk),'motion_span_pct':span}
        for name,d in [('current',a),('kalman',b)]:
            scores=[]
            for shift in range(-8,9):
                pred=np.array([d['frames'][k+shift][ident-1]['map'] for k in kk]);err=pred-ref;err-=err.mean(axis=0)
                scores.append(float(np.sqrt(np.mean(np.sum(err**2,axis=1)))))
            index=int(np.argmin(scores));contrast=(max(scores)-min(scores))/max(max(scores),1e-6)
            row[name]={'best_lag_frames':index-8,'best_lag_ms':(index-8)*40,'rms_by_lag':scores,'identifiable':index not in (0,16) and contrast>=.10}
        output.append(row)
    usable=[r for r in output if all(r[n]['identifiable'] for n in ('current','kalman'))]
    return {'scope':'Exploratory lag estimate against piecewise-linear interpolation of sparse reviewed floor points. Uses only original-ID-correct labeled endpoints and common map availability; constant position bias removed. Positive lag means output trails reference. Flat or boundary optima excluded. Not dense ground-truth timing.','segments':output,'usable_segments':len(usable),
      'current_median_lag_ms':float(np.median([r['current']['best_lag_ms'] for r in usable])) if usable else None,
      'kalman_median_lag_ms':float(np.median([r['kalman']['best_lag_ms'] for r in usable])) if usable else None,
      'median_added_lag_ms':float(np.median([r['kalman']['best_lag_ms']-r['current']['best_lag_ms'] for r in usable])) if usable else None}

def main():
    base=read(TRACK_DIR/'tracks.json')['roster'];labels=read(LABEL_DIR/'labels.json');baseline_hash=frozen_hash(base)
    configs=[dict(name=name+suffix,acceleration_sigma=q,measurement_sigma=r,measurement_median=median) for median,suffix in [(1,''),(3,'_median')] for name,q,r in [('smooth',12.,.6),('balanced',30.,.6),('responsive',60.,.45)]]
    results=[];runs=[]
    for cfg in configs:
        filtered=filter_maps(base,cfg);assert frozen_hash(filtered)==baseline_hash
        score=evaluate(filtered,labels,VAL);motion=motion_stats(base,filtered,start=165,end=196)
        results.append({'settings':cfg,'validation':score,'validation_motion':motion['metrics']});runs.append(filtered)
    # Preserve approximate validation accuracy, then minimize abrupt motion.
    before=evaluate(base,labels,VAL)
    eligible=[i for i,r in enumerate(results) if r['validation']['floor_correct_within_3pct']>=before['floor_correct_within_3pct']-2]
    chosen=min(eligible or range(len(results)),key=lambda i:(results[i]['validation_motion']['kalman']['p99_change_in_step_pct_width'],results[i]['validation']['median_floor_error_pct']))
    after=runs[chosen]
    write('selection.json',{'rule':'Iterative development: first tested three raw-input settings; added a three-observation median after raw-input filters increased map fluctuation. Among six settings, require validation correct-floor count within two of current, then minimize validation 99th-percentile change in map step; tie-break median position error. Four reviewed validation frames, motion window 165..195. Final section has already been reviewed; not a fresh independent benchmark.','selected':chosen,'candidates':results})
    write('settings.json',{**configs[chosen],'fps':25,'chi_square_gate':9.21,'persistent_change_frames':3,'max_prediction_frames':25,'max_correction_per_frame_pct_width':.875,'velocity_cap_pct_width_per_second':35})
    write('tracks.json',{'current':base,'kalman':after})
    scores={split:{name:evaluate(d,labels,frames) for name,d in [('current',base),('kalman',after)]} for split,frames in [('validation',VAL),('development_final',TEST)]}
    write('accuracy.json',scores);write('motion.json',motion_stats(base,after));write('stationary.json',stationary(base,after,labels));write('delay.json',delay_check(base,after,labels))
    write('provenance.json',{'frozen_boxes_ids_status_sha256':baseline_hash,'same_detector_boxes_ids_status':baseline_hash==frozen_hash(after),'input_tracks_sha256':hashlib.sha256((TRACK_DIR/'tracks.json').read_bytes()).hexdigest(),'source':str(SOURCE/'source.mp4'),'method':'Replaces only the map filter. Input is the raw bottom-center of frozen roster boxes; same observed/occluded eligibility. No detection, association, calibration or training changes. Causal; no future frames or reference points used.','label_limit':'Existing assistant-reviewed approximate annotations; no independent fresh holdout.'})
    print(json.dumps({'selected':configs[chosen],'scores':json.loads((OUT/'accuracy.json').read_text()),'delay':json.loads((OUT/'delay.json').read_text())['median_added_lag_ms']},indent=2),flush=True)

if __name__=='__main__':main()
