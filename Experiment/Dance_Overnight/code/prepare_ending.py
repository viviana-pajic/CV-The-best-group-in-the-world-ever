"""Additional development labels for seated poses and dim lighting; not a new test."""
from detector_probe import *
import prepare_expansion as review
OUT=ROOT/'Dance_Overnight'

def main():
    torch.set_num_threads(4);cv2.setNumThreads(2)
    folder=OUT/'data/ending';folder.mkdir(exist_ok=True)
    train=[5400,5420,5440,5480];val=[5510,5550]
    assert not (folder/'manual_points.txt').exists()
    model=YOLO(ROOT/'Dance YOLO trial/models/yolo26m.pt');proposals={}
    for k in train+val:
        im=cv2.imread(str(ROOT/'Dance_Project_Shared/dataset/frames'/f'frame_{k:06}.png'))
        rows=[o for o in merge(candidates(model,im,960,'0'),.65) if o['score']>=.1]
        rows.sort(key=lambda o:(o['box'][0]+o['box'][2])/2);proposals[f'frame_{k:06}.png']=rows
    write(folder/'proposals.json',proposals);review.OUT=folder;review.TRAIN=train;review.VAL=val;review.main(proposals)
    write(folder/'box_edits.json',{});write(folder/'extra_labels.json',{});(folder/'manual_points.txt').write_text('')
    write(folder/'annotation_scope.json',{'split':'Additional development only; correlated train/validation ending poses, no independent generalization claim','source':'Generic COCO box proposals. Support predictions not displayed during annotation.','purpose':'Test whether scarce seated and dim-light examples explain ending failures. Preserve both earlier frozen test results.'})

if __name__=='__main__':main()
