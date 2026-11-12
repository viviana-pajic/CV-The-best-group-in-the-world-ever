"""Serialize the authorized local GPU experiments; preserve logs and failures."""
import subprocess,sys,time,json,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Dance_Overnight';LOG=OUT/'logs';LOG.mkdir(exist_ok=True)
status=OUT/'gpu_queue_status.json';events=[]
def save():status.write_text(json.dumps(events,indent=2))
while not (OUT/'runs/v2_unbalanced_control/complete.json').exists():
    if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,11,10,6,15,tzinfo=datetime.timezone.utc):raise RuntimeError('Training wait exceeded overnight deadline')
    time.sleep(5)
jobs=[('support_v3',['train_support.py','--suite','v3']),('support_validation_v1',['candidate_evaluate.py','--pattern','v1_*']),('support_validation_v2',['candidate_evaluate.py','--pattern','v2_*']),('support_validation_v3',['candidate_evaluate.py','--pattern','v3_*']),('detector_posture',['detector_train.py','both'])]
for name,args in jobs:
    if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,11,10,6,15,tzinfo=datetime.timezone.utc):break
    start=time.time();event={'name':name,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'running'};events.append(event);save();print('START',name,flush=True)
    with (LOG/(name+'.log')).open('w',encoding='utf-8') as f:
        result=subprocess.run([sys.executable,str(OUT/'code'/args[0]),*args[1:]],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    event.update(status='complete' if result.returncode==0 else 'failed',exit_code=result.returncode,seconds=time.time()-start);save();print('DONE',name,result.returncode,flush=True)
print('GPU queue finished',flush=True)
