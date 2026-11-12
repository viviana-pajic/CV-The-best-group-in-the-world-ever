"""Concise supplemental evidence, separate from the original frozen evaluations."""
from pathlib import Path
import json,html
OUT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text())

def supplement():
    trained=sum((p/'complete.json').exists() for p in (OUT/'runs').iterdir() if p.is_dir())
    text=f'''## Later checks: more data, detection and visibility

We also added four seated/dim-light training frames and two development frames. The support set now has **60 training frames and 969 usable crops**, with **11 development frames and 180 clean crops**. There are {trained} completed support-training runs in the experiment folder. This is still a small, correlated dataset from one video.

The expanded model recovered **6/8 visible ending contacts**, compared with 0/8 for the previous model, using the same tracked boxes. On an earlier inspected section it improved from 105 to 111 correct positions out of 135 at 3%; on the later crowd section it dropped from 51 to 47 out of 77. These are later diagnostic replays with full forward context, not fresh heldout results. More examples helped the underrepresented seated pose; they did not fix the whole tracker.

The other checks were useful mainly for ruling things out:

- Synthetic contact occlusion did not beat the selected expanded model on real development images. Its synthetic success also benefits from retaining the original full-body box.
- Extending shortened crops did not improve visible-point accuracy and increased false visible reports for hidden contacts.
- Learned appearance features and re-identification recovery did not establish an improvement over the color/flow tracker. One proposed recovery remained unverified.
- Larger detector inputs and a larger generic detector did not improve the development box comparison.

The stationary-background audit found at most **1.66 source pixels of mat-anchor drift** across the extracted section. There is no meaningful ending zoom; lighting and crowding explain the appearance change. The fixed homography stays. Its assumed width/depth ratio is **1.6**, so map errors are relative and depend on that assumption. Pixel errors are saved alongside map errors.

Fresh CPU support inference reproduced the first frozen accuracy counts; its largest point difference from the cached GPU run was 0.07 source pixels. The cached final-filter reproduction also passed from the copied share folder. These checks validate the supplied inference path, not a claim that retraining always gives identical weights.
'''
    page='''<h2>More labels helped one weak posture</h2><p>The later support model uses 60 training images and 969 usable dancer crops. On two ending development images, it recovered 6 of 8 visible floor contacts through the actual tracking pipeline; the previous model recovered none. It improved one earlier diagnostic section but worsened another crowd section. This is a useful alternative, not an across-the-board win.</p><details><summary>Watch the expanded support model</summary><p>Same detector and IDs. The ending replay includes training frames; it is a diagnostic, not a fresh test.</p><video controls preload="none" poster="videos/expanded_support_ending_123.jpg" src="videos/expanded_support_ending.mp4"></video><video controls preload="none" src="videos/expanded_support_diagnostic.mp4"></video></details><h2>What did not earn a place in the pipeline</h2><p>Synthetic occlusion training, larger detector inputs, learned re-identification, and automatic crop extension did not demonstrate a reliable overall gain. Their results are retained. Hidden contacts stay unknown or explicitly estimated; the system does not invent measured floor points to keep twenty dots visible.</p><h2>Camera and reproduction checks</h2><p>Stationary-background alignment changed the mat anchors by at most 1.66 source pixels. The ending is darker, but there is no meaningful zoom. The mat transform remains fixed, with an assumed 1.6 width/depth ratio rather than surveyed dimensions.</p><p>Fresh CPU support inference reproduced the first frozen accuracy counts. The cached final-filter reproduction also passed from the copied share folder.</p>'''
    text+='\nThree additional YOLO fine-tunes used30 unique training frames, including four newly completed box-label images. The chosen brightness-augmentation run gave101 correct initial-ID matches and no wrong-ID matches among101 matched boxes out of120 development references. Its3%-correct support count stayed at74/77 using the same old support head. Detection improved modestly; it did not become a complete twenty-person tracker.\n'.replace('used30','used 30').replace('gave101','gave 101').replace('among101','among 101').replace('of120','of 120').replace('Its3%','Its 3%').replace('at74','at 74')
    confirmation=OUT/'final_confirmation/results.json'
    fallback=OUT/'fallback_experiment/results.json'
    if fallback.exists() and read(fallback)['selected'] is None:
        text+='\nThe old-first fallback kept106/112 correct development crop points but increased false visible reports from3 to5 compared with the expanded model. It was not retained.\n'.replace('kept106','kept 106').replace('from3','from 3').replace('to5','to 5')
        page+='<p>The old-first fallback combination also failed its development check: it kept the same correct-point count but reported more hidden contacts as visible.</p>'
    temporal=OUT/'temporal_experiment/results.json'
    if temporal.exists():
        paragraph='Three matched temporal fine-tunes compared the current image repeated three times with neighbouring images at ±80 ms and ±200 ms. Their development-selected configurations each placed 72/77 visible contacts within 3%, versus 74/77 for the retained ten-view single-image model. They did not justify replacing it. This is a small development comparison, not evidence that temporal learning cannot help.'
        text+='\n'+paragraph+'\n'
        page+='<p>'+html.escape(paragraph)+'</p>'
    if (OUT/'stationary_gap/results.json').exists():
        paragraph='Twelve cautious long-gap interpolation settings filled additional unknown track frames but left the controlled development accuracy unchanged. We have no reference coordinates behind the occlusions, so extra hollow dots cannot count as recovered dancers. These settings were not retained.'
        text+='\n'+paragraph+'\n';page+='<p>'+paragraph+'</p>'
    if (OUT/'diagnostics/latest_pipeline_sections.json').exists():
        paragraph='With the new detector as well as the expanded support model, the already-inspected full-context replays scored 108/135, 49/77 and 7/8 at 3%, versus 105/135, 51/77 and 0/8 for the established pipeline. The held-pose ID check matched 70/80 references correctly with the new detector, versus 73/80 before; neither had a wrong ID among matched boxes. Detection changes trade coverage across sections. Later crowded identities have not been independently audited.'
        text+='\n'+paragraph+'\n';page+='<p>'+paragraph+'</p>'
    if (OUT/'weak_contact/results.json').exists():
        paragraph='An exploratory follow-up adds only hollow estimates when a weak point is crop-stable and near a trusted temporal anchor. On the already inspected confirmation block it raised available visible positions from 34 to 38/46 and correct positions at 3% from 29 to 31/46, while hidden-contact estimates increased from 2 to 8/50. Hidden-position accuracy is unknown. On controlled development it improved 74 to 75/77 at 3%. This option is shown separately; it is not an independently confirmed replacement.'
        text+='\n### Optional estimates for short uncertain periods\n\n'+paragraph+'\n';page+='<h2>Optional estimates for uncertain periods</h2><p>'+paragraph+'</p><details><summary>Watch the extra hollow estimates</summary><video controls preload="none" src="videos/optional_estimates_development.mp4"></video><video controls preload="none" src="videos/optional_estimates_crowd.mp4"></video></details>'
    if (OUT/'geometry/results.json').exists():
        paragraph='A box-geometry regression baseline, trained on the same earlier 913 clean crops, placed 88/104 development contacts within 3% and reported 29 hidden contacts as visible. The image-based support model placed 99/104 correctly with 4 false visible reports, each at its development-selected threshold. This supports learning from the image rather than only correcting the box coordinates.'
        text+='\n'+paragraph+'\n';page+='<p>'+paragraph+'</p>'
    fresh=OUT/'verification/latest_fresh_inference/verification.json'
    if fresh.exists() and read(fresh)['passed']:
        paragraph='Fresh inference with the selected detector and expanded support weights exactly reproduced the saved boxes and support outputs on ten sampled source frames. A separate cached replay refit the full trajectory and reproduced the newest confirmation scores. Retraining reproducibility is a separate question.'
        text+='\n'+paragraph+'\n';page+='<p>'+paragraph+'</p>'
    repeat=OUT/'label_repeatability/results.json'
    seeds=OUT/'seed_repeatability/results.json'
    if seeds.exists():
        r=read(seeds)['models'];values={name:item['controlled']['correct_by_tolerance_pct_width']['3'] for name,item in r.items()}
        paragraph=f"Repeating the selected training recipe with seeds 29 and 43 gave {values['v9_repeat_wide_s29']}/77 and {values['v9_repeat_wide_s43']}/77 correct controlled-development positions at 3%, versus {values['v6_attention_wide_s17']}/77 for the original seed. The three-model average gave {values['ensemble']}/77 and requires three support-model passes. All used the fixed 0.25 visibility threshold and ten crop/flip views. This checks training variation on development data; it does not replace the independently confirmed candidate."
        text+='\n### Repeating the training recipe\n\n'+paragraph+'\n';page+='<h2>Repeating the training recipe</h2><p>'+paragraph+'</p>'
    efficiency=OUT/'efficiency/results.json'
    perturb=OUT/'box_perturbation/results.json'
    if perturb.exists():
        paragraph='With the image held fixed, eight ±4-pixel box perturbations were applied to each development crop. Across the 112 visible reference contacts, the expanded model’s point moved a median 0.36 source pixels (p90 1.73), versus 4 pixels for box bottoms. No map smoothing was used. This synthetic sensitivity check supports resistance to small box jitter; it does not test identity swaps or real movement. Hidden/uncertain predictions can still move much more.'
        text+='\n### Does the point resist box jitter?\n\n'+paragraph+'\n';page+='<h2>Does the point resist box jitter?</h2><p>'+paragraph+'</p>'
    if efficiency.exists():
        e=read(efficiency)['candidates']
        if len(e)==3:
            text+='\n### Accuracy and speed\n\nOn the same controlled development clip, one crop view gave 72/77 correct positions at 3%, two flip views gave 73/77, and ten crop/flip views gave 74/77. Warm support-head rates were respectively '+', '.join(f"{e[k]['warm_support_fps']:.1f}" for k in ['base','flip','crop5flip'])+' frames/s. These timings exclude detection, tracking and rendering. Keep ten views for the reported accuracy preset; two views are a promising faster development option, not an independently confirmed equivalent.\n'
            page+='<h2>A faster preview option</h2><p>Two flip views gave 73/77 correct development positions, versus 74/77 for ten crop/flip views. The support head alone ran at 16.0 versus 3.6 frames/s. These are development figures and exclude the detector, tracker and rendering; the reported frozen comparisons still use ten views.</p>'
    if repeat.exists():
        a=read(repeat)
        paragraph=f"A source-only repeat review agreed on contact status in {a['same_contact_status']}/20 crops. Among the {a['both_visible']} contacts marked visible both times, median placement difference was {a['median_pixel_difference']:.1f} source pixels ({a['median_stage_difference_pct_width']:.2f}% of mat width); the p90 difference was {a['p90_stage_difference_pct_width']:.2f}%. Three originally visible contacts were judged uncertain on the second pass. This is same-assistant repeatability, not independent human agreement. It reinforces the need for teammate review and limits how confidently we interpret small gains at the 1% tolerance. Frozen labels and scores were not changed."
        text+='\n### How precise are our labels?\n\n'+paragraph+'\n'
        page+='<div class="card warn"><strong>Review label precision before scaling up.</strong><p>'+html.escape(paragraph)+'</p></div>'
    if confirmation.exists():
        r=read(confirmation)
        text+='\n## Last confirmation block\n\n'+r['interpretation']+'\n\n'
        page+='<h2>Last confirmation block</h2><p>'+html.escape(r['interpretation'])+'</p>'
        headers=['Pipeline','Within 1%','Within 3%','Available','Hidden reported visible']
        rows=[]
        for name,d in r['candidates'].items():
            a=d['all'];n=a['visible_references'];rows.append([name,f"{a['correct_by_tolerance_pct_width']['1']}/{n}",f"{a['correct_by_tolerance_pct_width']['3']}/{n}",f"{a['available_visible']}/{n}",f"{a['hidden_measured']}/{a['hidden_references']}"])
        text+='| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
        page+='<div class="card scroll"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div><video controls preload="metadata" src="videos/final_confirmation.mp4"></video>'
    return text,page
