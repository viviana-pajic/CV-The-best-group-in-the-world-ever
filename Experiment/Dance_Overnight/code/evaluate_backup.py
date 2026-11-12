"""Final reserved comparison with fixed settings. Never select from these results."""
import frozen_holdout as fh
from frozen_holdout import *
import trajectory_fit

def main():
    dest=OUT/'backup_evaluation';dest.mkdir(exist_ok=True);fh.D=OUT/'data/backup';labels=read(fh.D/'labels.json');protocol=read(fh.D/'sealed_protocol.json');assert hashlib.sha256((fh.D/'labels.json').read_bytes()).hexdigest()==protocol['label_sha256'];base=read(OUT/'full_replay/forward_tracks.json');pred=read(OUT/'full_replay/support/predictions.json')[1170:];assert len(base['frames'])==len(pred);first=read(OUT/'search/frozen_selection.json')['offline']['chosen'];second=read(OUT/'round2/frozen_selection.json')['selection'];variants=[('raw_support',pred,first['threshold'],{'kind':'raw'},False),('round1',pred,first['threshold'],first['settings'],False),('round2',pred,second['threshold'],second['settings'],True)];box=copy.deepcopy(pred)
    for k,rr in enumerate(base['frames']):
        for o in rr:
            if str(o['id']) in box[k]:box[k][str(o['id'])]['point']=bottom(np.array(o['box'])).tolist()
    variants.append(('same_pipeline_box_bottom',box,second['threshold'],second['settings'],True));results={'protocol':protocol,'candidates':{}}
    for name,pp,threshold,settings,robust in variants:
        if settings['kind']=='raw':settings={**settings,'hold':0}
        z,cf,ids=inputs(base,pp,threshold);a,st,age=trajectory_fit.fit(z,cf,settings) if robust else filtered(z,cf,settings);tracks=export(base,a,st,age,z,ids,settings);write(dest/f'{name}_tracks.json',tracks);results['candidates'][name]={'end_to_end':fh.track_score(tracks,labels),'observed_only':fh.track_score(tracks,labels,observed_only=True),'threshold':threshold,'settings':settings};print('BACKUP',name,results['candidates'][name]['end_to_end']['all'],flush=True)
    write(dest/'results.json',results);print('Final reserved evaluation complete',flush=True)

if __name__=='__main__':main()
