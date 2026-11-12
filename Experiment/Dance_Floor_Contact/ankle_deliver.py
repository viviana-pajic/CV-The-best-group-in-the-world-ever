"""Show detected leg landmarks, accepted proxies, missing contacts, and stage maps."""
import json,subprocess,shutil
from collections import Counter
from pathlib import Path
import cv2,numpy as np
from ankle_pilot import OUT,LABEL_DIR,SOURCE,read,write,frozen_hash
from occlusion_deliver import source_frame,draw_map
from corrected_deliver import color
from render_controlled import text,BG,INK,MUTED

def draw_landmarks(im,row,threshold):
    im=im.copy()
    for o in row:
        pose=o.get('pose')
        if pose is None or o['status']!='observed':continue
        kp=np.array(pose['kp']);co=color(o['id'])
        for a,b in [(11,13),(13,15),(12,14),(14,16),(11,12)]:
            if min(kp[a,2],kp[b,2])>=threshold:
                cv2.line(im,tuple(kp[a,:2].astype(int)),tuple(kp[b,:2].astype(int)),co,2,cv2.LINE_AA)
        for j in [13,14,15,16]:
            if kp[j,2]<threshold:continue
            x,y=np.rint(kp[j,:2]).astype(int)
            if j>=15:cv2.circle(im,(x,y),4,(70,230,255),-1,cv2.LINE_AA)
            else:cv2.rectangle(im,(x-3,y-3),(x+3,y+3),(255,215,80),1)
        for j in range(17,min(len(kp),23)):
            if kp[j,2]>=threshold:cv2.circle(im,tuple(np.rint(kp[j,:2]).astype(int)),3,(100,255,100),-1,cv2.LINE_AA)
        if o.get('anchor_pixel') is not None:
            p=tuple(np.rint(o['anchor_pixel']).astype(int));cv2.drawMarker(im,p,(255,255,255),cv2.MARKER_CROSS,12,2,cv2.LINE_AA)
    return im

def render(data,labels,cfg,prefix='ankle'):
    specs={prefix+'_comparison':(25,0,299),'movement_half_speed':(12.5,0,149),'floor_work_half_speed':(12.5,225,299)}
    writers={n:cv2.VideoWriter(str(OUT/(n+'_raw.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),v[0],(1600,1000)) for n,v in specs.items()}
    cap=cv2.VideoCapture(str(SOURCE/'source.mp4'))
    method={'midpoint':'ANKLE MIDPOINT','ankle_support':'SUPPORT-ANKLE RULE','support':'ANKLES + KNEELING RULE','feet_midpoint':'TOE / HEEL MIDPOINT','feet_support':'TOE / HEEL SUPPORT','feet_knees':'FEET + KNEELING RULE'}[cfg['mode']]
    for k in range(300):
        ok,im=cap.read();assert ok
        canvas=np.full((1000,1600,3),BG,np.uint8)
        text(canvas,'FLOOR POSITION / BOX BOTTOM VS LEG LANDMARKS',24,36,.73,INK,2)
        text(canvas,f'+{k/25:05.2f}s   |   Same dancer boxes, IDs, calibration and smoothing   |   No retraining',24,65,.47,MUTED)
        for col,mode in enumerate(('current','ankles')):
            x=24+800*col;rows=data[mode]['frames'][k]
            text(canvas,'CURRENT / box-bottom floor proxy' if col==0 else 'POSE / '+method,x,101,.58,INK,2)
            if col:
                matched=sum(o.get('pose') is not None for o in rows);anchors=sum(o.get('anchor_pixel') is not None for o in rows)
                text(canvas,f'{matched} matched skeletons  |  {anchors} accepted raw support proposals',x,128,.43,MUTED)
            else:text(canvas,f"{sum(o['status']=='observed' for o in rows)} observed tracks  |  existing floor-point method",x,128,.43,MUTED)
            image=source_frame(im,rows,data[mode]['diagnostics'][k]['unassigned_high_confidence_boxes'])
            if col:image=draw_landmarks(image,rows,cfg['confidence'])
            canvas[160:395,x:x+752]=cv2.resize(image[275:675,:],(752,235))
            text(canvas,'CURRENT STAGE MAP' if col==0 else 'POSE STAGE MAP / no box-bottom fallback',x,447,.47,INK)
            cnt=draw_map(canvas,x+51,481,data[mode]['frames'],k,labels.get(str(k),[]))
            text(canvas,f"{cnt['proxy']} solid proxies   |   {cnt['estimated']} hollow estimates",x+51,932,.45,MUTED)
            if col:text(canvas,'Unknown floor locations: '+str(sum(o['map_status']=='unknown' for o in rows)),x+51,954,.40,MUTED)
            if col and k in [0,50,100,138,195,225,255,285,299]:
                text(image,'Yellow: ankles. Green: toes/heels (if available). Blue: knees. White: selected proxy.',12,690,.49,INK,1)
                cv2.imwrite(str(OUT/f'pose_detail_{k:03}.jpg'),image)
        text(canvas,'Landmarks: yellow = ankles; green = toes/heels (feet model); blue = knees; white = selected support proposal.',24,977,.42,MUTED)
        text(canvas,'Map: solid = filtered proxy; hollow = prediction <=1s; white crosses = reviewed references. Missing contacts stay unknown.',24,997,.40,MUTED)
        for name,(_,start,end) in specs.items():
            if start<=k<=end:writers[name].write(canvas)
        if k in [0,50,100,138,195,225,255,285,299]:cv2.imwrite(str(OUT/f'comparison_{k:03}.jpg'),canvas)
    cap.release()
    for w in writers.values():w.release()
    import imageio_ffmpeg
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    for name in specs:
        subprocess.run([ff,'-y','-i',str(OUT/(name+'_raw.mp4')),'-c:v','libx264','-crf','20','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/(name+'.mp4'))],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    print('Rendered ankle comparison videos',flush=True)

def report():
    accuracy=read(OUT/'accuracy.json');scores=accuracy['development_final'];a,b=scores['current'],scores['ankles'];cfg=read(OUT/'settings.json');coverage=read(OUT/'coverage.json');stat=read(OUT/'stationary.json');selection=read(OUT/'selection.json')
    improved=b['map']['floor_correct_within_3pct']>a['map']['floor_correct_within_3pct']
    verdict='More correct map positions on the reviewed final section.' if improved else 'Ankle landmarks do not improve overall map coverage and accuracy here.'
    fields=[('Raw correct positions / all visible references',f"{a['raw']['correct_within_3pct']} / 120",f"{b['raw']['correct_within_3pct']} / 120"),('Raw visible references with an accepted estimate',str(a['raw']['matched_visible_anchors']),str(b['raw']['matched_visible_anchors'])),('Median raw error (% of mat width)',f"{a['raw']['median_error_pct_width']:.2f}",f"{b['raw']['median_error_pct_width']:.2f}"),('Correct positions after identical smoothing / 120',str(a['map']['floor_correct_within_3pct']),str(b['map']['floor_correct_within_3pct'])),('Correct position AND original ID / 120',str(a['map']['floor_and_initial_id_correct']),str(b['map']['floor_and_initial_id_correct'])),('Correct original-ID matches / 145',str(a['map']['correct_initial_ids']),str(b['map']['correct_initial_ids']))]
    table=''.join(f'<tr><td>{n}</td><td>{x}</td><td>{y}</td></tr>' for n,x,y in fields)
    options=''.join(f"<tr><td>{r['settings']['mode']}</td><td>{r['settings']['confidence']}</td><td>{r['raw_validation']['correct_within_3pct']} / 48</td><td>{r['raw_validation']['emitted_anchors']}</td><td>{r['raw_validation']['joint_f1']:.3f}</td></tr>" for r in selection['candidates'])
    decision='Keep the ankle/support approach as a candidate, but it still needs checking on travel, overlap and floor work before broad labeling.' if improved else 'Keep the current map as the baseline. Do not ask teammates to label all ankles yet; this pretrained pose pipeline has not demonstrated a reliable overall improvement.'
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ankle floor-position pilot</title><style>
body{{background:#101619;color:#edf1f4;font:17px/1.65 system-ui;margin:0}}main{{max-width:1120px;margin:auto;padding:44px 24px 72px}}h1{{font-size:43px;line-height:1.12;margin:18px 0}}h2{{font-size:25px;margin-top:34px}}.tag{{color:#96d4c7;letter-spacing:2px;font-size:13px;text-transform:uppercase}}.card{{background:#1c282d;padding:24px;border-radius:14px;margin:24px 0}}video,img{{width:100%;border-radius:10px}}button{{padding:12px 16px;background:#293e47;color:#edf1f4;border:1px solid #526d79;border-radius:8px;margin:0 8px 12px 0;cursor:pointer}}a{{color:#9fcefa}}.muted{{color:#acbbc3}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #3c4d55}}th{{color:#96d4c7}}p{{max-width:940px}}</style><main>
<div class="tag">Dance project · floor-location experiment · 12-second clip</div><h1>Replace the floor point,<br>keep the tracking fixed.</h1><p><strong>{verdict}</strong> We used pretrained YOLO26m pose to locate leg joints, then changed only the point projected onto the mat. No new training labels or retraining were used.</p>
<div class="card"><button onclick="play('ankle_comparison.mp4')">Full clip · normal speed</button><button onclick="play('movement_half_speed.mp4')">First 6 seconds · half speed</button><button onclick="play('floor_work_half_speed.mp4')">Final floor work · half speed</button><video id="video" src="ankle_comparison.mp4" poster="comparison_195.jpg" controls playsinline preload="metadata"></video><p class="muted">Left: current box-bottom map. Right: selected pose method ({cfg['mode']}, joint confidence ≥ {cfg['confidence']}). Yellow circles are ankles; blue squares are knees; white crosses on the dancers are selected proposals. Solid map dots are filtered proxies, hollow dots are short predictions. There is no hidden fallback to the box bottom.</p></div>
<h2>What we tried</h2><p>Pose inference used three scene crops and enlarged crops around the existing detected dancers. Pose proposals were matched one-to-one to the frozen dancer boxes. The detector boxes, track identities and calibration did not change. Both maps use the existing median filter, bounded smoothing and maximum one-second hold.</p><p>We compared an ankle midpoint, an ankle rule that excludes an obviously raised foot, and a rule that can propose the knees for a bent-leg kneeling pose. Each was tested at two landmark-confidence thresholds. These geometry checks suggest contact; they do not prove it.</p>
<h2>Reviewed final-section results</h2><table><thead><tr><th>Check</th><th>Box bottom</th><th>Pose proposal</th></tr></thead><tbody>{table}</tbody></table><p class="muted">Eight previously reviewed frames contain 120 visible support references. A correct floor point is within 3% of mat width. Raw estimates are measured before smoothing. The all-reference counts penalize missing points; median error alone does not. Smoothed counts can include explicitly predicted points on observed dancers. Identity scores are identical by design.</p>
<h2>Availability across the full clip</h2><p>Across 6,000 dancer-frame records, the current map has {coverage['current'].get('proxy',0)} solid proxies, {coverage['current'].get('estimated',0)} predictions and {coverage['current'].get('unknown',0)} unknown locations. The pose map has {coverage['ankles'].get('proxy',0)} solid proxies, {coverage['ankles'].get('estimated',0)} predictions and {coverage['ankles'].get('unknown',0)} unknown locations. Twenty detected people does not imply twenty reliable floor contacts.</p>
<h2>Look at the landmarks</h2><img src="pose_detail_195.jpg" alt="Detected ankles and knees during raised-leg movement"><p class="muted">The model predicts ankle joints, not heels, toes or the sole touching the mat. A lifted ankle must not be directly treated as a point on the floor plane. Kneeling and overlapping bodies are particularly important checks.</p><img src="pose_detail_285.jpg" alt="Leg landmarks during floor work">
<h2>Validation choices</h2><table><thead><tr><th>Rule</th><th>Confidence</th><th>Correct / references</th><th>Emitted points</th><th>Joint F1</th></tr></thead><tbody>{options}</tbody></table><p class="muted">The rule maximizes raw-point joint F1 on the four existing validation frames, penalizing missed references and emitted wrong points. No final-section score was used to pick it. This remains iterative development on an already reviewed clip, not a fresh independent benchmark.</p>
<div class="card"><strong>Decision</strong><p>{decision}</p><p>If the project later needs targeted labels, distinguish the actual visible support point from ankle anatomy. Keep unknown or hidden contact unlabelled as a coordinate, and retain a visibility flag. Whole-person boxes and IDs are still useful for association. Segmentation alone does not identify which body part touches the floor.</p></div>
<h2>Limits</h2><p>The existing reference points are approximate assistant-reviewed labels. Some floor-work references describe support between knees, so an ankle midpoint may target a different point. Joint confidence is not a contact probability. The floor calibration is unchanged and uses relative dimensions. Current identity errors also remain; the pose stage cannot recover dancers that the frozen tracker marks unknown.</p>
<p><a href="Ankle_Pilot.ipynb">Concise notebook</a> · <a href="accuracy.json">Scores</a> · <a href="selection.json">All six settings</a> · <a href="coverage.json">Coverage</a> · <a href="provenance.json">Provenance</a> · <a href="https://docs.ultralytics.com/tasks/pose/">Official pose model and keypoint documentation</a></p>
</main><script>function play(src){{const v=document.getElementById('video');v.src=src;v.load();v.play();}}</script></html>'''
    (OUT/'START_HERE.html').write_text(body,encoding='utf-8')
    import nbformat as nbf
    nb=nbf.v4.new_notebook();nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}
    nb.cells=[nbf.v4.new_markdown_cell('# Ankle and support-point pilot\n\nCan a better floor-point input improve the map without changing detection or tracking? We used pretrained YOLO26m pose, with no additional labels or retraining.'),nbf.v4.new_code_cell("import json\nfrom pathlib import Path\nfrom IPython.display import Video, Image, display\nroot = Path.cwd()\nload = lambda name: json.loads((root / name).read_text())\nload('settings.json'), load('provenance.json')"),nbf.v4.new_markdown_cell('## 1. Locate joints, keep identities\n\nDetect the 17 COCO landmarks using scene tiles and enlarged dancer crops. Match pose proposals to existing boxes one-to-one. Keep all detector boxes and IDs fixed. COCO has ankle joints, not exact sole contacts.'),nbf.v4.new_code_cell("display(Image(filename=str(root / 'pose_detail_195.jpg')))"),nbf.v4.new_markdown_cell('## 2. Propose a floor point\n\nCompare an ankle midpoint, raised-foot rejection, and a kneeling-knee heuristic. Try two confidence thresholds. Select using the existing validation frames. Reject uncertain/covered landmarks. Keep the same map smoothing; never substitute a box bottom when the pose input fails.'),nbf.v4.new_code_cell("display(Video(str(root / 'ankle_comparison.mp4'), embed=False, width=1000))"),nbf.v4.new_markdown_cell('## 3. Check accuracy and missing estimates together\n\nMedian error on the surviving points can look good even when many dancers disappear. Compare correct raw and filtered positions against all 120 visible final-section references, and report coverage separately.'),nbf.v4.new_code_cell("results = load('accuracy.json')['development_final']\nfor name, values in results.items():\n    print(name, 'raw correct:', values['raw']['correct_within_3pct'], 'filtered correct:', values['map']['floor_correct_within_3pct'], '/ 120')\nload('coverage.json')"),nbf.v4.new_markdown_cell('## 4. Decision\n\n'+verdict+' '+decision+'\n\nThis is an already reviewed development clip with approximate support labels. Ankle joints and ground-contact points are different targets, especially during raised-leg movement and kneeling. Identity errors and calibration are unchanged.\n\nReproduction: from the DL project root, run `Dance_Floor_Contact/ankle_pilot.py infer`, then `ankle_pilot.py` and `ankle_deliver.py` in the existing Python environment. The source clip, prior frozen tracking artifacts and model weights are required.\n\n[Official YOLO pose documentation](https://docs.ultralytics.com/tasks/pose/)')]
    from nbclient import NotebookClient
    NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(OUT)}}).execute();nbf.write(nb,OUT/'Ankle_Pilot.ipynb')
    for name in ['ankle_pilot.py','ankle_deliver.py']:shutil.copy2(Path(__file__).parent/name,OUT/name)

def verify(data,prefix='ankle'):
    assert frozen_hash(data['current'])==frozen_hash(data['ankles'])
    for row in data['ankles']['frames']:
        for o in row:
            if o['map_status']=='estimated':assert 0<o['map_age_frames']<=25
            if o['map_status']=='unknown':assert o['map'] is None
            if o.get('anchor_pixel') is not None:assert o.get('pose') is not None
    results={'same_boxes_ids_status':True}
    for name,count,fps in [(prefix+'_comparison',300,25),('movement_half_speed',150,12.5),('floor_work_half_speed',75,12.5)]:
        cap=cv2.VideoCapture(str(OUT/(name+'.mp4')));actual=cap.get(cv2.CAP_PROP_FPS);n=0
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape==(1000,1600,3);n+=1
        cap.release();assert n==count and actual==fps;results[name]={'decoded_frames':n,'fps':actual}
    write('verification.json',results);print(json.dumps(results),flush=True)

if __name__=='__main__':
    data=read(OUT/'tracks.json');labels=read(LABEL_DIR/'labels.json');cfg=read(OUT/'settings.json')
    render(data,labels,cfg);report();verify(data)
