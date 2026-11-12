"""Reproduce the latest frozen confirmation from supplied tracking/support caches."""
from pathlib import Path
import argparse
import frozen_holdout as fh
from frozen_holdout import *
import trajectory_fit
from evaluate_final_confirmation import score

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args();dest=Path(args.output);assert not dest.exists(),'Choose a new output folder';dest.mkdir(parents=True);cfg=read(OUT/'final_confirmation/frozen_selection.json');provenance=read(OUT/'late_full_replay/provenance.json');base=read(OUT/provenance['base_tracks']);pred=read(OUT/provenance['support_predictions']);z,cf,ids=inputs(base,pred,cfg['threshold']);a,st,age=trajectory_fit.fit(z,cf,cfg['settings']);tracks=export(base,a,st,age,z,ids,cfg['settings']);fh.D=OUT/'data/final_confirmation';labels=read(fh.D/'labels.json');actual=score(tracks,labels);expected=read(OUT/'final_confirmation/results.json')['candidates']['Expanded support'];keys=['visible_references','available_visible','correct_by_tolerance_pct_width','hidden_measured','hidden_estimated','matched_visible_boxes']
    for key in keys:assert actual['all'][key]==expected['all'][key],(key,actual['all'][key],expected['all'][key])
    for key in ['median_error_pct','p90_error_pct','median_source_pixel_error']:assert abs(actual['all'][key]-expected['all'][key])<1e-7,key
    write(dest/'tracks.json',tracks);write(dest/'results.json',actual);write(dest/'verification.json',{'passed':True,'scope':'New robust map fit from supplied frozen detector/ID and support caches; full2424-frame context. No fresh detector/support inference claimed. Matches latest confirmation scores exactly.','counts':actual['all']['correct_by_tolerance_pct_width'],'reference_file':'final_confirmation/results.json','selection_file':'final_confirmation/frozen_selection.json'});print('Latest confirmation reproduced exactly:',actual['all']['correct_by_tolerance_pct_width'],flush=True)

if __name__=='__main__':main()
