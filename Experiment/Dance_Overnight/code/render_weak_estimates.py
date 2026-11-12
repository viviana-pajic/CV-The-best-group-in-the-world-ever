"""Show extra estimates separately; no change to the frozen accuracy videos."""
from render_comparison import render,read,OUT,ROOT,cv2
import copy

def section(data,a,b):
    d=copy.deepcopy(data);d['frames']=d['frames'][a:b]
    if 'diagnostics' in d:d['diagnostics']=d['diagnostics'][a:b]
    return d

if __name__=='__main__':
    cv2.setNumThreads(2);base=read(OUT/'late_full_replay/tracks.json');weak=read(OUT/'weak_contact/tracks.json');pred=read(OUT/'late_full_replay/support/predictions.json');folder=OUT/'weak_contact'
    for name,first,last in [('development',4420,4719),('crowd',3950,4080)]:
        a,b=first-3250,last-3250+1;source=folder/(name+'.avi');cap=cv2.VideoCapture(str(OUT/'full_replay/source.avi'));cap.set(cv2.CAP_PROP_POS_FRAMES,a);writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'FFV1'),25,(1280,720))
        for k in range(a,b):ok,im=cap.read();assert ok;writer.write(im)
        cap.release();writer.release();render(section(base,a,b),section(weak,a,b),source,OUT/'videos'/f'optional_estimates_{name}.mp4',pred=pred[a:b],before_pred=pred[a:b],threshold=.25,before_threshold=.25,source_first=first,title='OPTIONAL ESTIMATES / EXPLORATORY REPLAY',subtitle='Inspected data, not a fresh test | Same detector and IDs',before_title='Current conservative map',after_title='Add bounded weak-contact estimates',ring_description='Extra hollow dots are estimates, not recovered measurements. Hidden-position accuracy is unknown.')
