"""Reproduce final reserved map filtering and scores in a new directory."""
import argparse
from pathlib import Path
import frozen_holdout as fh
from frozen_holdout import *
import trajectory_fit

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(OUT/'reproduced_final'));args=p.parse_args();dest=Path(args.output);assert not dest.exists(),'Use a new output directory';dest.mkdir(parents=True);cfg=read(OUT/'round2/frozen_selection.json')['selection'];base=read(OUT/'full_replay/forward_tracks.json');pred=read(OUT/'full_replay/support/predictions.json')[1170:];z,cf,ids=inputs(base,pred,cfg['threshold']);a,st,age=trajectory_fit.fit(z,cf,cfg['settings']);tracks=export(base,a,st,age,z,ids,cfg['settings']);write(dest/'tracks.json',tracks);fh.D=OUT/'data/backup';score=fh.track_score(tracks,read(fh.D/'labels.json'));saved=read(OUT/'backup_evaluation/results.json')['candidates']['round2']['end_to_end'];assert score==saved,'Reproduced scores differ; inspect the saved inputs and environment';write(dest/'results.json',score);print('Final reserved scores reproduced exactly from cached detections/support predictions. No training or fresh model inference performed.')

if __name__=='__main__':main()
