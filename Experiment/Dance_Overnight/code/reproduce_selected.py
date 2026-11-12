"""Rerun the selected support model and map filter with preserved detector tracks.

Example: python code/reproduce_selected.py --device cpu --output reproduction_01
Detection/tracking are held fixed; this is a fresh support-head inference run.
"""
import argparse,time
from pathlib import Path
from frozen_holdout import *

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['causal','offline'],default='offline');args=p.parse_args();out=args.output.resolve()
    if out.exists():raise FileExistsError('Choose a new output directory to preserve earlier results.')
    torch.set_num_threads(4);cv2.setNumThreads(2);chosen=read(OUT/'search/frozen_selection.json')[args.mode]['chosen'];base=read(OUT/'reserved_evaluation'/args.mode/'base_tracks.json');start=time.perf_counter();pred=predict_pipeline(chosen['support'],base,out,args.device);z,cf,ids=inputs(base,pred,chosen['threshold']);a,st,age=filtered(z,cf,chosen['settings']);tracks=export(base,a,st,age,z,ids,chosen['settings']);write(out/'tracks.json',tracks)
    reference=read(OUT/'reserved_evaluation'/args.mode/'predictions.json');diff=[]
    for aa,bb in zip(pred,reference):
        assert set(aa)==set(bb)
        diff.extend(float(np.linalg.norm(np.array(aa[k]['point'])-np.array(bb[k]['point']))) for k in aa)
    summary={'mode':args.mode,'device':args.device,'seconds':time.perf_counter()-start,'frames':len(pred),'median_source_pixel_difference_from_saved_inference':float(np.median(diff)),'maximum_source_pixel_difference_from_saved_inference':float(np.max(diff)),'reserved_scores':track_score(tracks,read(D/'labels.json'))['all'],'scope':'Fresh support-head inference and map filtering, using frozen detector/identity tracks. Floating-point differences across hardware are expected.'};write(out/'verification.json',summary);print(summary,flush=True)

if __name__=='__main__':main()
