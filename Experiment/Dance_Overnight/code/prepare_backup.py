"""Final small reserved sample, prepared only after round2 was frozen."""
from detector_probe import *
import prepare_expansion as review
OUT=ROOT/'Dance_Overnight'

def main():
    torch.set_num_threads(4);cv2.setNumThreads(2);assert (OUT/'round2/frozen_selection.json').exists();folder=OUT/'data/backup';folder.mkdir(exist_ok=True);frames=[5125,5145,5165,5195,5225,5245,5265];model=YOLO(ROOT/'Dance YOLO trial/models/yolo26m.pt');proposals={}
    assert not (folder/'manual_points.txt').exists()
    for k in frames:
        im=cv2.imread(str(ROOT/'Dance_Project_Shared/dataset/frames'/f'frame_{k:06}.png'));rows=[o for o in merge(candidates(model,im,960,'0'),.65) if o['score']>=.1];rows.sort(key=lambda o:(o['box'][0]+o['box'][2])/2);proposals[f'frame_{k:06}.png']=rows
    write(folder/'proposals.json',proposals);review.OUT=folder;review.TRAIN=[];review.VAL=frames;review.main(proposals);write(folder/'box_edits.json',{});write(folder/'extra_labels.json',{});(folder/'manual_points.txt').write_text('');write(folder/'annotation_scope.json',{'split':'Final reserved evaluation only','source':'Generic COCO proposals, no support predictions displayed','selection':'Both comparison variants frozen before these images were inspected. No training/threshold/filter selection on this sample.'})

if __name__=='__main__':main()
