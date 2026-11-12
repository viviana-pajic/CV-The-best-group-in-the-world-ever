"""Finish evaluation and videos after the full-sequence inference completes."""
from render_comparison import *
import time

def main():
    while not (OUT/'full_replay/provenance.json').exists():time.sleep(5)
    subprocess.run([sys.executable,str(OUT/'code/evaluate_backup.py')],check=True)
    cfg=read(OUT/'round2/frozen_selection.json')['selection'];base=read(OUT/'backup_evaluation/same_pipeline_box_bottom_tracks.json');after=read(OUT/'backup_evaluation/round2_tracks.json');first=5125;last=5265;source=OUT/'videos/backup_source.avi';cap=cv2.VideoCapture(str(OUT/'full_replay/source.avi'));cap.set(cv2.CAP_PROP_POS_FRAMES,first-3250);w=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'FFV1'),25,(1280,720));assert w.isOpened()
    for k in range(last-first+1):ok,im=cap.read();assert ok;w.write(im)
    cap.release();w.release()
    for d in [base,after]:
        for key in ['frames','diagnostics']:d[key]=d[key][first-4420:last-4420+1]
    labels={str(int(k)-first):v for k,v in read(OUT/'data/backup/labels.json').items()};pred=read(OUT/'full_replay/support/predictions.json')[first-3250:last-3250+1]
    render(base,after,source,OUT/'videos/final_reserved.mp4',labels,pred,cfg['threshold'],first,'FINAL RESERVED SECTION / FLOOR POINT COMPARISON','Identical boxes, IDs, visibility and offline trajectory fit','Learned floor support',before_title='Box-bottom support')
    base=read(OUT/'full_replay/same_detector_box_baseline.json');after=read(OUT/'full_replay/tracks.json');pred=read(OUT/'full_replay/support/predictions.json')
    render(base,after,OUT/'full_replay/source.avi',OUT/'videos/full_section.mp4',{},pred,cfg['threshold'],3250,'FULL EXTRACTED SECTION / TRACKING STRESS TEST','Offline | One initialization at176.8s | Includes training frames','Learned support + robust trajectory fit',before_title='Same detector / previous box map')
    print('Full replay deliverables complete',flush=True)

if __name__=='__main__':main()
