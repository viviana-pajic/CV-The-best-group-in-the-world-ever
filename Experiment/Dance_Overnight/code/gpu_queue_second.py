"""Second research batch, serialized after the initial detector/support queue."""
import subprocess,sys,time,json,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Dance_Overnight';LOG=OUT/'logs'
events=[];status=OUT/'gpu_queue_second_status.json'
def save():status.write_text(json.dumps(events,indent=2))
deadline=datetime.datetime(2026,11,10,6,15,tzinfo=datetime.timezone.utc)
while True:
    if datetime.datetime.now(datetime.timezone.utc)>=deadline:raise RuntimeError('Deadline reached before second queue')
    try:
        old=json.loads((OUT/'gpu_queue_status.json').read_text())
        if len(old)==5 and all(o['status']!='running' for o in old):break
    except (OSError,json.JSONDecodeError):pass
    time.sleep(5)
jobs=[('support_v4',['train_support.py','--suite','v4']),('support_v5',['train_support.py','--suite','v5']),('support_validation_v4',['candidate_evaluate.py','--pattern','v4_*']),('support_validation_v5',['candidate_evaluate.py','--pattern','v5_*'])]
for name,args in jobs:
    if datetime.datetime.now(datetime.timezone.utc)>=deadline:break
    event={'name':name,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'running'};events.append(event);save();start=time.time();print('START',name,flush=True)
    with (LOG/(name+'.log')).open('w',encoding='utf-8') as f:
        result=subprocess.run([sys.executable,str(OUT/'code'/args[0]),*args[1:]],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    event.update(status='complete' if result.returncode==0 else 'failed',exit_code=result.returncode,seconds=time.time()-start);save();print('DONE',name,result.returncode,flush=True)
print('Second queue finished',flush=True)
