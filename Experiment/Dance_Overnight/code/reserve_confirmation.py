"""Reserve a final temporal block before viewing its labels or candidate outputs."""
from pathlib import Path
import json,datetime
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Dance_Overnight';D=OUT/'final_confirmation'
FRAMES=[3953,3977,3999,4021,4043,4061,4079]
if __name__=='__main__':
    D.mkdir(exist_ok=True);assert not (D/'precommit.json').exists()
    manifests=['development_v1.json','development_v2.json','development_expanded.json','development_ending.json'];seen=set()
    for name in manifests:
        data=json.loads((OUT/'data'/name).read_text())
        seen.update(d['source_first']+k for d in data.values() for split in ['train','val'] for k in d[split])
    assert not set(FRAMES)&seen
    (D/'precommit.json').write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_frames':FRAMES,'temporal_block':[3950,4080],'known_development_overlap':[],'protocol':'Do not inspect candidate outputs or create contact annotations in this block until the late candidate choices are frozen. Then annotate source-only generic proposals, seal labels, evaluate the frozen and late pipelines once, and preserve all results. No tuning on this confirmation set.','scope':'One choreography, assistant labels, image-local IDs. Full-video inference has already processed this block, but its outputs and contact labels have not been inspected for selecting the late candidates. It is a new labeled confirmation block, not a separate video or independently human-audited benchmark.'},indent=2));print('Reserved final confirmation frames',FRAMES)
