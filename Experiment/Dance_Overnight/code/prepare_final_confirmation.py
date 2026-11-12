"""Source-only proposal sheets, created only after all late choices are frozen."""
from detector_probe import *
import prepare_expansion as review

def main():
    out=ROOT/'Dance_Overnight';d=out/'data/final_confirmation'
    while not (out/'final_confirmation/frozen_selection.json').exists():time.sleep(5)
    while not (out/'efficiency/results.json').exists() or len(read(out/'efficiency/results.json')['candidates'])<3:time.sleep(5)
    assert not (d/'manual_points.txt').exists();pre=read(out/'final_confirmation/precommit.json');frames=pre['source_frames'];manifest=read(out/'data/development_ending.json');used={x['source_first']+k for x in manifest.values() for split in ['train','val'] for k in x[split]};assert not any(k in used for k in range(*[pre['temporal_block'][0],pre['temporal_block'][1]+1]))
    torch.set_num_threads(3);cv2.setNumThreads(2);d.mkdir(exist_ok=True);model=YOLO(ROOT/'Dance YOLO trial/models/yolo26m.pt');proposals={}
    for k in frames:
        im=cv2.imread(str(ROOT/'Dance_Project_Shared/dataset/frames'/f'frame_{k:06}.png'));rows=[o for o in merge(candidates(model,im,960,'cpu'),.65) if o['score']>=.1];rows.sort(key=lambda o:(o['box'][0]+o['box'][2])/2);proposals[f'frame_{k:06}.png']=rows
    write(d/'proposals.json',proposals);review.OUT=d;review.TRAIN=[];review.VAL=frames;review.main(proposals);write(d/'box_edits.json',{});write(d/'extra_labels.json',{});(d/'manual_points.txt').write_text('');write(d/'annotation_scope.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'New confirmation block. Generic source-image box proposals only, no support outputs. Frozen selection exists before preparation. No training/validation source frame anywhere inside3950..4080. Near neighbors remain in the same choreography.','selection_sha256':hashlib.sha256((out/'final_confirmation/frozen_selection.json').read_bytes()).hexdigest()})

if __name__=='__main__':
    import datetime,hashlib
    main()
