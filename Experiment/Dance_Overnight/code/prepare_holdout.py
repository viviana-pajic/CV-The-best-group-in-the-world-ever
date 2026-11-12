"""Reserved evaluation annotation sheets; generic detector boxes, no floor predictions."""
from detector_probe import *
import prepare_expansion as review

def main():
    torch.set_num_threads(4);cv2.setNumThreads(2);folder=ROOT/'Dance_Overnight/data/holdout';folder.mkdir(exist_ok=True);plan=read(ROOT/'Dance_Overnight/search_freeze.json')['reserved_evaluation'];frames=plan['label_frames'];model=YOLO(ROOT/'Dance YOLO trial/models/yolo26m.pt');proposals={}
    assert not (folder/'manual_points.txt').exists(),'Do not overwrite an in-progress annotation set'
    for k in frames:
        im=cv2.imread(str(ROOT/'Dance_Project_Shared/dataset/frames'/f'frame_{k:06}.png'));rows=[o for o in merge(candidates(model,im,960,'0'),.65) if o['score']>=.1];rows.sort(key=lambda o:(o['box'][0]+o['box'][2])/2);proposals[f'frame_{k:06}.png']=rows
    write(folder/'proposals.json',proposals);review.OUT=folder;review.TRAIN=[];review.VAL=frames;review.main(proposals);write(folder/'box_edits.json',{});write(folder/'extra_labels.json',{});(folder/'manual_points.txt').write_text('');write(folder/'annotation_scope.json',{'split':'reserved evaluation only','source':'Generic COCO YOLO26m proposals, no choreography-tuned model; all contacts blank. Boxes and completeness require manual review.','annotation_protocol':plan['labels'],'no_training_or_selection':True})

if __name__=='__main__':main()
