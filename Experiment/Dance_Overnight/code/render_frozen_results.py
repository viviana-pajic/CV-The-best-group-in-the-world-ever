"""Side-by-side development and reserved comparisons for the frozen pipeline."""
import copy
from render_comparison import *

def main():
    selection=read(OUT/'search/frozen_selection.json');dest=OUT/'videos';dest.mkdir(exist_ok=True);long=OUT/'long_sequence';source=dest/'reserved_source.avi'
    if not source.exists():
        cap=cv2.VideoCapture(str(long/'source.avi'));cap.set(cv2.CAP_PROP_POS_FRAMES,380);writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'FFV1'),25,(1280,720));assert writer.isOpened()
        for k in range(281):ok,im=cap.read();assert ok;writer.write(im)
        cap.release();writer.release()
    labels={str(int(k)-4800):rows for k,rows in read(OUT/'data/holdout/labels.json').items()};base=read(long/'reference_tracks.json');baseline=copy.deepcopy(base);baseline['frames']=base['frames'][380:];baseline['diagnostics']=base['diagnostics'][380:]
    for mode in ['offline','causal']:
        chosen=selection[mode]['chosen'];data=read(OUT/'reserved_evaluation'/mode/'tracks.json');data['frames']=data['frames'][380:];data['diagnostics']=data['diagnostics'][380:];pred=read(OUT/'reserved_evaluation'/mode/'predictions.json')[380:]
        subtitle='Frozen settings | Full-clip future information used' if mode=='offline' else 'Frozen settings | Past and current frames only in map filter'
        # A bidirectional tracker itself uses future frames even with a causal map filter.
        if mode=='causal' and chosen['tracker']=='bidir':subtitle='Causal map filter | Tracker uses full-clip future information'
        render(baseline,data,source,dest/f'reserved_{mode}.mp4',labels,pred,chosen['threshold'],4800,'RESERVED SECTION / BEFORE AND AFTER',subtitle,'Learned support + '+mode+' filtering')
    chosen=selection['offline']['chosen'];base=read(ROOT/'Dance_Tracking_Pilot/tracks.json')['roster'];data=read(OUT/chosen['folder']/'evaluation/offline_tracks.json');pred=read(OUT/chosen['folder']/'predictions.json');render(base,data,ROOT/'Dance_Controlled_Test/source.mp4',dest/'development_offline.mp4',read(ROOT/'Dance_Support_Training_Pilot/labels.json'),pred,chosen['threshold'],4420,'DEVELOPMENT SECTION / BEFORE AND AFTER','Used for tuning | Full-clip future information used','Learned support + offline filtering')
    print('Three comparisons rendered and decoded successfully',flush=True)

if __name__=='__main__':main()
