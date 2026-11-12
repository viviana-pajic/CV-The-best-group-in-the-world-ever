from pathlib import Path
import json,datetime
OUT=Path(__file__).resolve().parents[1]
status={
 'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'state':'Latest frozen candidate improves the new floor-work confirmation; temporal-context development trials and final handoff QA remain in progress.',
 'completed':[
  'Five detector fine-tunes; selected diverse_brightness trained on30 unique frames',
  'Expanded support data:60 training frames/969 usable crops;11 validation frames/180 clean crops',
  'New confirmation block3950..4080:103 manually reviewed distinguishable bodies,46 visible contacts,50 hidden,7 uncertain',
  'Fresh frozen confirmation: expanded29/46 at3%, incumbent25/46, same-pipeline box bottom19/46; hidden measured5/50 versus8/50 incumbent',
  'New full2424-frame replay, two131-frame confirmation videos, twelve-second clearer development demo',
  'Source-only repeat annotation:17/20 status agreement;14 both-visible points differ6.3 pixels median; three ambiguous contacts flagged',
  'CVAT main review:71 images,60train split20/20/20 plus11validation; XML roundtrip1247 boxes/803 points passed',
  'Latest cached confirmation reproduced exactly from full sequence caches',
  'Old-first support fallback failed development eligibility; no gain retained'],
 'active':['Five-frame temporal-context support trial, then matched-control development comparison','Fresh selected detector/support inference verification after temporal queue','Final report/notebook refresh, portable copies/ZIPs, browser/video/link QA'],
 'deadline_utc':'2026-11-10T07:00:00Z',
 'evaluation_status':'All three frozen evaluation rounds stay outside training. Later temporal and fallback experiments select only on development; they do not silently replace the frozen confirmation candidate.',
 'selected_config':'final_confirmation/frozen_selection.json',
 'package_status':'Earlier draft share folders/ZIP exist and must be refreshed after final results. No CVAT cloud import or Drive upload performed overnight.',
 'limitations':'Not twenty reliable dancers throughout. New confirmation does not audit identity switches. Same-video, approximate assistant labels, incomplete crowd ground truth, assumed mat aspect1.6, offline future-frame filtering.'}
(OUT/'CURRENT_STATUS.json').write_text(json.dumps(status,indent=2))
print(status['state'])
