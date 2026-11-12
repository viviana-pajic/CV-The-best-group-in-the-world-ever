"""Build a portable, local handoff. No cloud upload or messages are performed."""
from pathlib import Path
import argparse,json,shutil,hashlib,zipfile,datetime
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Dance_Overnight'
DEST=ROOT/'Dance_Team_Shared_2026-11-10';EXPERIMENT=DEST/'Experiment'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def allowed(p):
    rel=p.relative_to(OUT);parts=rel.parts;name=p.name
    if name=='CURRENT_STATUS.json' or name.endswith('_queue_status.json') or name=='gpu_queue_second_status.json':return False
    if '__pycache__' in parts or p.suffix in ['.pyc','.log','.tmp']:return False
    if name.endswith('_raw.mp4') or name in ['merged_color.json','raw_color.json']:return False
    if parts[0]=='verification':return name in ['verification.json','results.json']
    if parts[0]=='videos':return p.suffix in ['.jpg','.json'] or p.suffix=='.mp4' and not name.endswith('_raw.mp4')
    if parts[0]=='tracking':return p.suffix=='.json' and ('selection' in name or 'results' in name)
    if parts[0]=='recovery':return p.suffix=='.json' and 'results' in name
    if parts[0]=='search' and len(parts)>1 and p.suffix=='.json':return not name.endswith('_tracks.json') and name not in ['tracks.json','predictions.json','base_tracks.json']
    if parts[0]=='long_sequence' and (p.suffix=='.npz' or name.endswith('features.json')):return False
    if parts[0]=='detector':
        if p.suffix=='.pt':return rel.as_posix()=='detector/runs/posture_frozen/weights/best.pt'
        if name.endswith('features.json') or name.endswith('_tracks.json'):return False
    if parts[0]=='detector_expanded':
        if p.suffix=='.pt':
            choice=OUT/'detector_expanded/tracking_selection.json'
            return choice.exists() and rel.as_posix()==f"detector_expanded/runs/{read(choice)['chosen']['model']}/weights/best.pt"
        if name.endswith('features.json') or name.endswith('_tracks.json'):return False
    if parts[0]=='runs' and p.suffix=='.pt':
        keep={('v5_attention_s17','best_joint.pt'),('v5_mask_s17','best_joint.pt')}
        for run in ['v9_repeat_wide_s29','v9_repeat_wide_s43']:keep.add((run,'best_joint.pt'))
        for run in ['v7_occlusion_p0.3_s17','v7_occlusion_p0.6_s17']:
            keep.update({(run,'best_joint.pt'),(run,'best_masked.pt')})
        temporal=OUT/'temporal_experiment/results.json'
        if temporal.exists():
            for run,row in read(temporal)['runs'].items():
                if row.get('chosen'):keep.add((run,row['chosen']['candidate']['checkpoint']))
        supplementary=OUT/'ending_experiment/results.json'
        if supplementary.exists() and read(supplementary).get('chosen'):
            c=read(supplementary)['chosen']['candidate'];keep.add((c['run'],c['checkpoint']))
        return (parts[1],name) in keep
    if parts[0]=='reid':return p.suffix=='.json' and ('results' in name or name in ['complete.json','history.json'])
    if parts[0]=='reid_full':return name in ['diagnostic_results.json','provenance.json']
    if parts[0]=='external' and p.suffix=='.pt':return False
    if p.suffix=='.npz':return False
    return p.suffix in ['.py','.json','.html','.txt','.ipynb','.png','.jpg','.mp4','.avi','.pt','.csv','.yaml','.xml','.zip','.md'] or name=='LICENSE'

def copy_file(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.stat().st_size==source.stat().st_size and target.stat().st_mtime_ns==source.stat().st_mtime_ns:return
    shutil.copy2(source,target)

def build():
    assert DEST.resolve().parent==ROOT.resolve()
    previous=read(DEST/'PACKAGE_MANIFEST.json')['files'] if (DEST/'PACKAGE_MANIFEST.json').exists() else []
    for item in previous:
        rel=Path(item['path']);prefix=Path('Experiment/Dance_Overnight')
        if not rel.is_relative_to(prefix):continue
        source=OUT/rel.relative_to(prefix);target=(DEST/rel).resolve()
        if source.exists() and not allowed(source) and target.is_relative_to(DEST.resolve()) and target.is_file():
            with target.open('rb') as f:unchanged=hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']
            if unchanged:target.unlink() # Only an unchanged generated copy; the source is preserved.
    for p in OUT.rglob('*'):
        if p.is_file() and allowed(p):
            target=EXPERIMENT/'Dance_Overnight'/p.relative_to(OUT)
            # An audit marker must not cause retraining to skip missing weights.
            if p.name=='complete.json' and ('runs' in p.parts) and not any(allowed(x) for x in p.parent.glob('*.pt')) and not any(allowed(x) for x in (p.parent/'weights').glob('*.pt')):target=target.with_name('audit_complete.json')
            copy_file(p,target)
    for p in (ROOT/'Dance_Floor_Contact').glob('*.py'):copy_file(p,EXPERIMENT/'Dance_Floor_Contact'/p.name)
    fixed=['Dance_CVAT_Pilot/calibration_draft.json','Dance_Controlled_Test/source.mp4','Dance_Corrected_Pilot/labels.json','Dance_Corrected_Pilot/predictions_detect.json','Dance_Corrected_Pilot/settings.json','Dance_Support_Training_Pilot/labels.json','Dance_Tracking_Pilot/tracks.json','Dance_Tracking_Pilot/settings.json','Dance YOLO trial/models/yolo26m.pt']
    for rel in fixed:
        p=ROOT/rel
        if p.exists():copy_file(p,EXPERIMENT/rel)
    frames=set()
    for manifest in ['development_v1.json','development_v2.json','development_expanded.json','development_ending.json']:
        p=OUT/'data'/manifest
        if not p.exists():continue
        for d in read(p).values():frames.update(d['source_first']+k for split in ['train','val'] for k in d[split])
    dp=read(OUT/'detector/dataset_provenance.json');frames.update(dp['source_first']+k for k in dp['train_frames']+dp['validation_frames'])
    if (OUT/'temporal_experiment/results.json').exists():
        for d in read(OUT/'data/development_ending.json').values():
            for split in ['train','val']:
                for k in d[split]:frames.update([d['source_first']+k+x for x in [-5,-2,2,5]])
    for k in frames:
        rel=Path('Dance_Project_Shared/dataset/frames')/f'frame_{k:06}.png';copy_file(ROOT/rel,EXPERIMENT/rel)
    (DEST/'START_HERE.html').write_text('''<!doctype html><meta charset="utf-8"><title>Dance project handoff</title><style>body{font:20px/1.65 system-ui;max-width:850px;margin:70px auto;padding:20px;background:#f4f3ee;color:#213c34}a{color:#16745e}</style><h1>Dance stage mapping</h1><p><a href="Experiment/Dance_Overnight/START_HERE.html">Open the results, videos and notebook</a></p><p><a href="Experiment/Dance_Overnight/CVAT_Review/GUIDE.html">Open the teammate labeling guide</a></p><p>Download and unzip the whole folder before opening these pages. Google Drive stores the files; the HTML report and videos work locally. Keep the Experiment folders together so the reproduction scripts can find their inputs.</p><p>The report distinguishes frozen evaluation from later diagnostic experiments. These are assistant-prepared labels for human review.</p>''',encoding='utf-8')
    (DEST/'CONTENTS.txt').write_text('Local share package. No upload has been performed.\n\nSTART_HERE.html opens the report.\nExperiment/Dance_Overnight contains the notebook, videos, selected models, source clips, labels and reproducible scripts.\nSibling helper folders preserve the original relative paths.\nUnselected weights, dense optical flow and redundant raw video encodes are omitted.\nOmitted runs retain audit_complete.json and training histories.\nOpen Experiment/Dance_Overnight/REPRODUCE.txt for the supported reproduction commands.\n\nAll reference labels are approximate assistant visual estimates. Reserved evaluation images must remain separate from training.\n',encoding='utf-8')
    files=[]
    for p in sorted(DEST.rglob('*')):
        if p.is_file() and p.name!='PACKAGE_MANIFEST.json':
            with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
            files.append({'path':p.relative_to(DEST).as_posix(),'bytes':p.stat().st_size,'sha256':digest})
    (DEST/'PACKAGE_MANIFEST.json').write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Portable local experiment and team review; no cloud transmission','files':files},indent=2))
    print('Packaged',len(files),'files',round(sum(x['bytes'] for x in files)/1e9,3),'GB',flush=True)

def archive(folder,path):
    media={'.mp4','.avi','.png','.jpg','.zip','.pt'}
    with zipfile.ZipFile(path,'w',allowZip64=True,compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file():z.write(p,(Path(folder.name)/p.relative_to(folder)).as_posix(),compress_type=zipfile.ZIP_STORED if p.suffix in media else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(path) as z:assert z.testzip() is None
    print('Archive verified',path.name,round(path.stat().st_size/1e6,1),'MB',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--zip',action='store_true');a=parser.parse_args();build()
    if a.zip:archive(DEST,ROOT/(DEST.name+'.zip'))
