"""Precommit late selection and confirmation rules before inspecting its labels."""
from pathlib import Path
import json,datetime,hashlib
OUT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text())
if __name__=='__main__':
    dest=OUT/'final_confirmation/selection_protocol.json'
    assert not dest.exists()
    support=read(OUT/'occlusion_candidates/results.json')['chosen']
    cfg={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'support':support,
         'support_rule':'Real development joint F1 with older-development F1 within 0.02 of V5 reference at the same threshold. Synthetic masked scores do not select the model.',
         'detector_rule':'Compare the development-selected expanded detector against original frozen detector using identical support head and robust map filter. Promote detector only if correct identity matches are at least100/120, wrong identity matches at most1, and 3%-correct support count is no lower than the original detector on the same controlled validation frames. Otherwise retain original detector.',
         'filter':read(OUT/'round2/frozen_selection.json')['selection']['settings'],
         'confirmation_variants':['incumbent V5 point model and original detector','development-selected V6 point model and development-approved detector','same selected pipeline with box-bottom point substitution'],
         'confirmation_decision':'Report all three. Call the late pipeline an overall improvement on this block only if its 3%-correct count is at least the incumbent, its1%-correct count improves, and hidden contacts emitted as measured increase by no more than1. If these conditions fail, retain the incumbent as the established reference and describe the late model as a posture-specific alternative. Never tune a threshold or filter from this block.',
         'scope':'One new labeled block in the same choreography, not independent human ground truth. Image-local IDs cannot evaluate identity switches. Assumed mat aspect1.6 unchanged. Missing visible points count as misses.',
         'support_checkpoint_sha256':hashlib.sha256((OUT/'runs'/support['candidate']['run']/support['candidate']['checkpoint']).read_bytes()).hexdigest()}
    dest.write_text(json.dumps(cfg,indent=2))
    print('Late selection protocol saved; confirmation labels still absent.')
