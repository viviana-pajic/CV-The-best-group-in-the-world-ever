"""Clearer twelve-second development demo of the latest frozen pipeline."""
from render_comparison import *
from render_final_confirmation import subset

if __name__=='__main__':
    p=read(OUT/'late_full_replay/provenance.json');cfg=read(OUT/'final_confirmation/frozen_selection.json');tracks=read(OUT/p['tracks']);box=read(OUT/'final_confirmation/box_bottom_full_tracks.json');pred=read(OUT/p['support_predictions']);labels=read(ROOT/'Dance_Support_Training_Pilot/labels.json')
    render(subset(box,4420,4719),subset(tracks,4420,4719),ROOT/'Dance_Controlled_Test/source.mp4',OUT/'videos/latest_development_demo.mp4',labels,pred[1170:1470],cfg['threshold'],4420,'CLEARER SECTION / DEVELOPMENT DEMO','Includes training frames | Same detector, IDs and offline filter','Learned floor support',before_title='Box-bottom floor point')
