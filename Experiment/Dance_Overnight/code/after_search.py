"""Bounded queued evaluation; no parameter changes based on reserved results."""
import time,datetime,subprocess,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Dance_Overnight'
deadline=datetime.datetime(2026,11,10,6,15,tzinfo=datetime.timezone.utc)
while not (OUT/'search/selection_ready.json').exists():
    if datetime.datetime.now(datetime.timezone.utc)>=deadline:raise RuntimeError('Morning evaluation deadline reached')
    time.sleep(10)
for name in ['frozen_holdout','filter_characterize','render_frozen_results']:
    print('START',name,flush=True)
    with (OUT/'logs'/f'{name}.log').open('w') as f:subprocess.run([sys.executable,str(OUT/'code'/f'{name}.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    print('DONE',name,flush=True)
print('Frozen evaluation and previews complete',flush=True)
