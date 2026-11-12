"""Finish the precommitted support/tracking/filter search on development data only."""
import time,datetime,subprocess,sys,copy,hashlib,shutil
from pipeline_infer import infer,read,write,ROOT,OUT,torch,cv2,np
from pipeline_eval import run as evaluate
from compare import stage

TRACKERS={
 'flow':OUT/'tracking/flow_b1.0_gap1_a0.5_tracks.json',
 'original':ROOT/'Dance_Tracking_Pilot/tracks.json',
 'bidir':OUT/'tracking/bidirectional_union_tracks.json',
 'frozen':OUT/'detector/sequences/posture_frozen_960/posture_frozen_h0.15_l0.06_a0.7_tracks.json',
}
KEYS=['run','checkpoint','mode','decoder']
SEARCH=OUT/'search'

def same_cfg(a,b):return all(a.get(k)==b.get(k) for k in KEYS)
def same_tracks(a,b):
    return len(a['frames'])==len(b['frames']) and all(x['id']==y['id'] and x['status']==y['status'] and x['box']==y['box'] for xx,yy in zip(a['frames'],b['frames']) for x,y in zip(xx,yy))

def ensemble(a,b,mode):
    rows=[]
    for aa,bb in zip(a,b):
        result={}
        assert set(aa)==set(bb)
        for ident,p in aa.items():
            q=bb[ident];x=np.array(p['point']);y=np.array(q['point']);u=p['reportability'];v=q['reportability'];score=(u+v)/2
            point=(x+y)/2
            if mode=='disagreement':
                point=(u*x+v*y)/max(u+v,1e-8);score*=np.exp(-.5*(np.linalg.norm(stage(x)-stage(y))/3)**2)
            result[ident]={'point':point.tolist(),'reportability':float(score),'model_disagreement_pct':float(np.linalg.norm(stage(x)-stage(y)))}
        rows.append(result)
    return rows

def choose(records,mode):
    pool=[r for r in records if r['mode']==mode];top=max(r['accuracy']['correct_by_tolerance_pct_width']['3'] for r in pool)
    near=[r for r in pool if r['accuracy']['correct_by_tolerance_pct_width']['3']>=top-1]
    safe=lambda r:r['accuracy']['hidden_measured']<=2 and (r['continuity']['candidate']['acceleration_p99'] or 0)<=.8
    eligible=[r for r in near if safe(r)];reason='Within1 of best3%-correct count, at most2 hidden measured, acceleration p99<=0.8.'
    if not eligible:
        eligible=[r for r in pool if safe(r)]
        if eligible:
            best=max(r['accuracy']['correct_by_tolerance_pct_width']['3'] for r in eligible);eligible=[r for r in eligible if r['accuracy']['correct_by_tolerance_pct_width']['3']>=best-1]
            reason='No near-best option met visibility/continuity limits; prioritize the best safe subset and disclose accuracy tradeoff.'
        else:
            eligible=near;reason='No pipeline met both visibility/continuity limits; choose near-best accuracy and flag limitations.'
    c=min(eligible,key=lambda r:(-r['accuracy']['correct_by_tolerance_pct_width']['1'],r['accuracy']['p90_error_pct'],r['complexity']))
    return {'chosen':c,'rule_applied':reason,'best_3pct_count_overall':top}

def main():
    torch.set_num_threads(4);cv2.setNumThreads(2);SEARCH.mkdir(exist_ok=True)
    while True:
        try:status=read(OUT/'gpu_queue_second_status.json')
        except (ValueError,OSError):time.sleep(2);continue
        if len(status)==4 and all(r['status']=='complete' for r in status):break
        if any(r['status']=='failed' for r in status):raise RuntimeError('Training/validation queue failed')
        if datetime.datetime.now(datetime.timezone.utc).hour>=5:raise RuntimeError('Too late to start full search')
        time.sleep(10)
    subprocess.run([sys.executable,str(OUT/'code/rescore_candidates.py')],check=True,stdout=subprocess.DEVNULL)
    ranking=read(OUT/'candidates/clean_ranking.json');cost={'base':1,'flip':2,'crop5':5,'crop5flip':10}
    ranking.sort(key=lambda r:(-r['validation']['selected']['joint_f1'],-r['validation']['selected']['correct'],cost[r['mode']]))
    shortlist=[];seen=set()
    for r in ranking:
        key=(r['run'],r['epoch'])
        if key not in seen:shortlist.append(r);seen.add(key)
        if len(shortlist)==3:break
    champion=read(OUT/'pipelines/v2_flow/settings.json')
    if not any(same_cfg(champion,r) for r in shortlist):shortlist.append(champion)
    write(SEARCH/'shortlist.json',shortlist)
    oldfolders=[p for p in (OUT/'pipelines').iterdir() if (p/'settings.json').exists() and (p/'predictions.json').exists()];catalog=[]
    for i,cfg in enumerate(shortlist):
        for tracker,path in TRACKERS.items():
            folder=SEARCH/f's{i+1}_{tracker}';base=read(path);base=base.get('roster',base);folder.mkdir(exist_ok=True)
            if not (folder/'predictions.json').exists():
                sources=[p for p in oldfolders if same_cfg(cfg,read(p/'settings.json'))]
                exact=next((p for p in sources if same_tracks(base,read(p/'base_tracks.json'))),None)
                if exact:
                    for name in ['predictions.json','base_tracks.json','timing.json']:shutil.copy2(exact/name,folder/name)
                    write(folder/'settings.json',cfg)
                    if (exact/'evaluation/results.json').exists():shutil.copytree(exact/'evaluation',folder/'evaluation',dirs_exist_ok=True)
                    print('Reused exact development pipeline',folder.name,flush=True)
                else:
                    reuse=SEARCH/f's{i+1}_flow' if tracker!='flow' else (sources[0] if sources else None)
                    infer(cfg,base,ROOT/'Dance_Controlled_Test/source.mp4',folder,'cuda',reuse)
            if not (folder/'evaluation/results.json').exists():evaluate(folder)
            catalog.append({'name':folder.name,'folder':str(folder.relative_to(OUT)),'support':cfg,'tracker':tracker,'complexity':cost[cfg['mode']]+{'flow':0,'original':0,'bidir':1,'frozen':0}[tracker]})
    for mode in ['plain','disagreement']:
        for tracker in TRACKERS:
            a=SEARCH/f's1_{tracker}';b=SEARCH/f's2_{tracker}';folder=SEARCH/f'ensemble_{mode}_{tracker}';folder.mkdir(exist_ok=True)
            cfg={'type':'ensemble','ensemble_mode':mode,'members':shortlist[:2]}
            if not (folder/'predictions.json').exists():
                write(folder/'predictions.json',ensemble(read(a/'predictions.json'),read(b/'predictions.json'),mode));shutil.copy2(a/'base_tracks.json',folder/'base_tracks.json');write(folder/'settings.json',cfg)
            if not (folder/'evaluation/results.json').exists():evaluate(folder)
            catalog.append({'name':folder.name,'folder':str(folder.relative_to(OUT)),'support':cfg,'tracker':tracker,'complexity':cost[shortlist[0]['mode']]+cost[shortlist[1]['mode']]+{'flow':0,'original':0,'bidir':1,'frozen':0}[tracker]})
    records=[]
    for meta in catalog:
        result=read(OUT/meta['folder']/'evaluation/results.json')
        for mode,r in result['chosen'].items():records.append({**meta,'mode':mode,'threshold':result['threshold'],**r})
    write(SEARCH/'pipeline_ranking.json',records)
    selection={mode:choose(records,mode) for mode in ['causal','offline']}
    selection['scope']='Development-only selection; no reserved floor predictions or scores used.';selection['frozen_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    selection['hashes']={}
    for mode in ['causal','offline']:
        cfg=selection[mode]['chosen']['support'];members=cfg.get('members',[cfg])
        for r in members:
            path=OUT/'runs'/r['run']/r['checkpoint'];selection['hashes'][str(path.relative_to(OUT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    write(SEARCH/'frozen_selection.json',selection)
    write(SEARCH/'selection_ready.json',{'utc':selection['frozen_utc'],'selection_sha256':hashlib.sha256((SEARCH/'frozen_selection.json').read_bytes()).hexdigest()})
    for mode in ['causal','offline']:
        r=selection[mode]['chosen'];print('FROZEN',mode,r['name'],r['accuracy']['correct_by_tolerance_pct_width'],'hidden',r['accuracy']['hidden_measured'],flush=True)

if __name__=='__main__':main()
