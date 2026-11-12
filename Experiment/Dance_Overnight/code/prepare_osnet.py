"""Retrieve the author's OSNet code and MSMT17 ReID weights; preserve source/license hashes."""
from pathlib import Path
import requests,hashlib,json,ast
OUT=Path(__file__).resolve().parents[1];D=OUT/'external/osnet';D.mkdir(parents=True,exist_ok=True)
repo=requests.get('https://api.github.com/repos/KaiyangZhou/deep-person-reid/commits/master',timeout=30);repo.raise_for_status();commit=repo.json()['sha'];meta=requests.get('https://huggingface.co/api/models/kaiyangzhou/osnet',timeout=30);meta.raise_for_status();info=meta.json();filename=next(r['rfilename'] for r in info['siblings'] if r['rfilename'].startswith('osnet_x1_0_msmt17_combineall'))
urls={'osnet.py':f'https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/{commit}/torchreid/models/osnet.py','LICENSE':f'https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/{commit}/LICENSE','reid_weights.pt':f"https://huggingface.co/kaiyangzhou/osnet/resolve/{info['sha']}/{filename}"};records={}
for name,url in urls.items():
    p=D/name
    if not p.exists():
        response=requests.get(url,timeout=120);response.raise_for_status();p.write_bytes(response.content)
    records[name]={'url':url,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
tree=ast.parse((D/'osnet.py').read_text());print('Source imports:',[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]);(D/'provenance.json').write_text(json.dumps(records,indent=2));print('Downloaded official ReID model',filename)
