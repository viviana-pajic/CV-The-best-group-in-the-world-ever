"""Bounded, explicitly estimated long gaps when body tracks remain nearly stationary."""
from frozen_holdout import *
import itertools

def fill(data,max_gap=150,max_distance=2.,max_residual=.2):
    out=copy.deepcopy(data);ids=sorted({o['id'] for row in data['frames'] for o in row});index=[{o['id']:o for o in row} for row in out['frames']];events=[];n=len(index)
    for ident in ids:
        valid=np.array([ident in row and row[ident]['map'] is not None for row in index]);k=0
        while k<n:
            if valid[k]:k+=1;continue
            start=k
            while k<n and not valid[k]:k+=1
            end=k
            if start==0 or end==n or end-start>max_gap:continue
            left=index[start-1][ident];right=index[end][ident]
            if left['map_status']!='proxy' or right['map_status']!='proxy':continue
            before=np.array(left['map']);after=np.array(right['map'])
            if np.linalg.norm(after-before)>max_distance:continue
            rows=[index[j].get(ident) for j in range(start-1,end+1)]
            if any(o is None for o in rows) or np.mean([o['status']=='observed' for o in rows])<.8:continue
            boxes=np.array([o['box'] for o in rows]);centers=(boxes[:,:2]+boxes[:,2:])/2;u=np.linspace(0,1,len(rows))[:,None];linear=(1-u)*centers[0]+u*centers[-1];residual=np.linalg.norm(centers-linear,axis=1)/np.median(boxes[:,3]-boxes[:,1])
            if np.max(residual)>max_residual:continue
            events.append({'id':ident,'first':start,'last':end-1,'frames':end-start,'endpoint_distance_pct_width':float(np.linalg.norm(after-before)),'max_body_center_residual_in_body_heights':float(max(residual))})
            for j in range(start,end):
                t=(j-start+1)/(end-start+1);o=index[j][ident];age=min(j-start+1,end-j);o.update(map=((1-t)*before+t*after).tolist(),map_status='estimated',map_age_frames=age,uncertainty=.5+.1*age,map_estimate_reason='Offline interpolation between measured endpoints; small body-track motion. No hidden-contact measurement.')
    out['stationary_gap_estimates']={'max_gap_frames':max_gap,'max_endpoint_distance_pct_width':max_distance,'max_body_residual_heights':max_residual,'events':events,'scope':'Only bounded unknown intervals are filled. Requires measured endpoints and at least80% observed body tracks. Does not prove identity continuity or hidden-position accuracy.'};return out

def main():
    dest=OUT/'stationary_gap';dest.mkdir(exist_ok=True);full=read(OUT/'late_full_replay/tracks.json');base=copy.deepcopy(full)
    for key in ['frames','diagnostics']:base[key]=base[key][1170:1470]
    d=read(OUT/'data/development_v1.json')['controlled'];ids=sorted({o['id'] for row in base['frames'] for o in row});a,st=original_arrays(base,ids);baseline=identity_score(a,st,d['labels'],d['val'],ids);rows=[]
    for gap,distance,residual in itertools.product([75,150,300],[2.,4.],[.2,.35]):
        filled=fill(base,gap,distance,residual);a,st=original_arrays(filled,ids);score=identity_score(a,st,d['labels'],d['val'],ids);rows.append({'settings':{'max_gap':gap,'max_distance':distance,'max_residual':residual},'development':score,'estimated_track_frames_added':sum(x['frames'] for x in filled['stationary_gap_estimates']['events'])})
    eligible=[r for r in rows if r['development']['correct_by_tolerance_pct_width']['3']>baseline['correct_by_tolerance_pct_width']['3'] and r['development']['p90_error_pct']<=baseline['p90_error_pct'] and r['development']['hidden_measured']<=baseline['hidden_measured']]
    selected=max(eligible,key=lambda r:(r['development']['correct_by_tolerance_pct_width']['3'],-r['estimated_track_frames_added'])) if eligible else None;write(dest/'results.json',{'scope':'Development-only display/estimation experiment. Hidden contact coordinates have no ground truth; extra hollow dots must not be described as recovered measured positions. No change to frozen confirmation.','baseline':baseline,'candidates':rows,'selected':selected});print('STATIONARY GAP baseline',baseline['correct_by_tolerance_pct_width'],'selected',selected is not None,flush=True)
    for r in rows:print(r['settings'],r['development']['correct_by_tolerance_pct_width'],r['estimated_track_frames_added'],flush=True)
    if selected:write(dest/'full_tracks.json',fill(full,**selected['settings']))

if __name__=='__main__':main()
