"""Render all precommitted confirmation comparisons and the longer diagnostic replay."""
from render_comparison import *
import time,copy

def subset(data,first,last):
    d=copy.deepcopy(data)
    for key in ['frames','diagnostics']:d[key]=d[key][first-3250:last-3250+1]
    return d

def main():
    while not (OUT/'final_confirmation/results.json').exists():time.sleep(5)
    cfg=read(OUT/'final_confirmation/frozen_selection.json');p=read(OUT/'late_full_replay/provenance.json');old=read(OUT/'ending_experiment/pipeline/previous_support_full_tracks.json');new=read(OUT/p['tracks']);box=read(OUT/'final_confirmation/box_bottom_full_tracks.json');oldpred=read(OUT/'full_replay/support/predictions.json');newpred=read(OUT/p['support_predictions']);oldthreshold=read(OUT/'round2/frozen_selection.json')['selection']['threshold'];first,last=3950,4080;source=OUT/'videos/final_confirmation_source.avi';cap=cv2.VideoCapture(str(OUT/'full_replay/source.avi'));cap.set(cv2.CAP_PROP_POS_FRAMES,first-3250);w=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'FFV1'),25,(1280,720));assert w.isOpened()
    for k in range(last-first+1):ok,im=cap.read();assert ok;w.write(im)
    cap.release();w.release();labels={str(int(k)-first):v for k,v in read(OUT/'data/final_confirmation/labels.json').items()};op=oldpred[first-3250:last-3250+1];npred=newpred[first-3250:last-3250+1]
    render(subset(old,first,last),subset(new,first,last),source,OUT/'videos/final_confirmation.mp4',labels,npred,cfg['threshold'],first,'NEW CONFIRMATION / FROZEN BEFORE LABELING','New detector + expanded support | Same mat and offline filter','Expanded candidate',before_title='Established reference',before_pred=op,before_threshold=oldthreshold,ring_description='Rings: raw support predictions. Map dots: offline fitted positions. White crosses: approximate source-only reference.')
    render(subset(box,first,last),subset(new,first,last),source,OUT/'videos/final_confirmation_point_ablation.mp4',labels,npred,cfg['threshold'],first,'NEW CONFIRMATION / POINT ABLATION','Same detector, IDs, visibility and offline filter','Learned floor support',before_title='Box bottom')
    render(old,new,OUT/'full_replay/source.avi',OUT/'videos/late_full_section.mp4',None,newpred,cfg['threshold'],3250,'FULL SECTION / LATE CANDIDATE DIAGNOSTIC','Includes training images | One initialization at176.8s | Offline','Expanded detector + support',before_title='Established detector + support',before_pred=oldpred,before_threshold=oldthreshold,ring_description='Rings: raw support predictions. Map dots: offline fitted positions. This replay includes training images.')

if __name__=='__main__':main()
