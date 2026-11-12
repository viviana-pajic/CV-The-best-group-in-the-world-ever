"""Check the real handoff: local links, executable notebook outputs and inputs."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import json,hashlib,argparse

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ['href','src','poster'] and value:self.links.append(value)

def check(root):
    out=root/'Dance_Overnight' if (root/'Dance_Overnight').exists() else root
    result={'html_pages':{},'notebooks':{},'selected_model_files':{},'training_frames':0}
    for rel in ['START_HERE.html','CVAT_Review/GUIDE.html']:
        page=out/rel;parser=Links();parser.feed(page.read_text(encoding='utf-8'));local=[]
        for link in parser.links:
            url=urlsplit(link)
            if url.scheme or url.netloc or not url.path:continue
            target=(page.parent/unquote(url.path)).resolve();assert target.exists(),(page,link);local.append(link)
        result['html_pages'][rel]={'local_links_checked':len(local)}
    for name in ['Dance_Research_Notebook.ipynb','Experiment_History.ipynb']:
        nb=json.loads((out/name).read_text(encoding='utf-8'));codes=[c for c in nb['cells'] if c['cell_type']=='code'];errors=[o for c in codes for o in c['outputs'] if o['output_type']=='error'];assert not errors,(name,errors);assert all(c['execution_count'] is not None for c in codes),name
        result['notebooks'][name]={'executed_code_cells':len(codes),'errors':len(errors)}
    cfg=json.loads((out/'final_confirmation/frozen_selection.json').read_text());weights=[out/cfg['detector']['weights'],out/'runs'/cfg['support']['run']/cfg['support']['checkpoint']]
    for p in weights:
        assert p.exists(),p
        with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        result['selected_model_files'][str(p.relative_to(out))]=digest
    manifest=json.loads((out/'data/development_ending.json').read_text());frames={d['source_first']+k for d in manifest.values() for split in ['train','val'] for k in d[split]}
    for k in frames:assert (out.parent/'Dance_Project_Shared/dataset/frames'/f'frame_{k:06}.png').exists(),k
    result['training_frames']=len(frames);result['passed']=True
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=check(a.root);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
