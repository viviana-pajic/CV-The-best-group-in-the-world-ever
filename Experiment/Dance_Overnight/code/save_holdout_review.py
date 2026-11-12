"""Record source-image visual review; no model contact coordinates used."""
import json
from pathlib import Path
D=Path(__file__).resolve().parents[1]/'data/holdout'
def read(n):return json.loads((D/n).read_text())
def write(n,x):(D/n).write_text(json.dumps(x,indent=2))
rows={
5080:'F497,468 H D H H F573,467 H H F638,473 H D H F709,476 F764,464 F811,475',
}
text=(D/'manual_points.txt').read_text()
for k,v in rows.items():
    assert not any(x.startswith(str(k)+':') for x in text.splitlines()),k
    text+=f'{k}: {v}\n'
(D/'manual_points.txt').write_text(text)
edits=read('box_edits.json');edits['5080']={'10':[636,377,680,444]}
write('box_edits.json',edits)
extra=read('extra_labels.json')
def row(i,b,contact='hidden'):
    return dict(id=i,box=b,floor=None,contact=contact,support_type=None,outside=False,reviewed=True,provenance='Assistant source-image visual addition separating overlapping dancers; image-local ID, no model point used.')
extra['5080']=[dict(row(101,[1131,447,1207,528],'visible'),floor=[1173,520],support_type='feet')]
write('extra_labels.json',extra)
print('Saved rows and explicit box corrections')
