"""A concise review package for the trained floor-support pilot."""
import json,subprocess,shutil,zipfile,sys
from pathlib import Path
from collections import Counter
import cv2,numpy as np
from support_evaluate import OUT,write
from corrected_setup import SOURCE,read
from support_label_setup import TRAIN,VAL,TEST,FRAMES
from occlusion_deliver import source_frame,draw_map
from render_controlled import text,BG,INK,MUTED
from map_kalman_pilot import frozen_hash

def render(data,labels,diagnostic=False):
    specs={'point_only_comparison':(25,0,299),'point_only_test_half_speed':(12.5,225,299)} if diagnostic else {'support_comparison':(25,0,299),'test_half_speed':(12.5,225,299)}
    writers={n:cv2.VideoWriter(str(OUT/(n+'_raw.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),v[0],(1600,1000)) for n,v in specs.items()}
    cap=cv2.VideoCapture(str(SOURCE/'source.mp4'))
    for k in range(300):
        ok,im=cap.read();assert ok
        canvas=np.full((1000,1600,3),BG,np.uint8)
        text(canvas,'LEARNING THE FLOOR-SUPPORT POINT',24,36,.80,INK,2)
        section='TRAIN PERIOD' if k<=132 else 'VALIDATION PERIOD' if 165<=k<=195 else 'TEST PERIOD' if k>=225 else 'UNLABELLED GAP'
        text(canvas,f'+{k/25:05.2f}s  |  {section}  |  Same detector, dancer IDs and stage calibration',24,65,.48,MUTED)
        for col,mode in enumerate(['current','point_only' if diagnostic else 'learned']):
            x=24+800*col;rows=data[mode]['frames'][k]
            text(canvas,'BEFORE / box bottom' if col==0 else ('DIAGNOSTIC / support point only' if diagnostic else 'AFTER / trained support + visibility'),x,101,.59,INK,2)
            accepted=sum(o.get('anchor_pixel') is not None for o in rows)
            text(canvas,f"{sum(o['status']=='observed' for o in rows)} observed dancers  |  {accepted} raw point proposals",x,128,.44,MUTED)
            annotated=source_frame(im,rows,data[mode]['diagnostics'][k]['unassigned_high_confidence_boxes'])
            for o in rows:
                if o.get('anchor_pixel') is not None:
                    p=tuple(np.rint(o['anchor_pixel']).astype(int));cv2.circle(annotated,p,5,(90,235,130) if col else (0,170,255),2,cv2.LINE_AA)
            for g in labels.get(str(k),[]):
                if g['floor'] is not None:cv2.drawMarker(annotated,tuple(g['floor']),(255,255,255),cv2.MARKER_CROSS,9,1,cv2.LINE_AA)
            canvas[160:395,x:x+752]=cv2.resize(annotated[275:675,:],(752,235))
            text(canvas,'BOX-BOTTOM STAGE MAP' if col==0 else 'LEARNED SUPPORT STAGE MAP',x,447,.49,INK)
            cnt=draw_map(canvas,x+51,481,data[mode]['frames'],k,labels.get(str(k),[]))
            text(canvas,f"{cnt['proxy']} measured proxies  |  {cnt['estimated']} held estimates  |  {sum(o['map_status']=='unknown' for o in rows)} unknown",x+42,937,.42,MUTED)
            if col and k in [0,165,225,245,265,285,299]:cv2.imwrite(str(OUT/f'{"point_only_" if diagnostic else ""}detail_{k:03}.jpg'),annotated)
        text(canvas,'Orange = box bottom. Green = learned support proposal. White crosses = fresh visual reference points.',24,975,.45,MUTED)
        text(canvas,'DIAGNOSTIC: visibility rejection disabled. Points on hidden contacts can be wrong.' if diagnostic else 'Solid map dot = filtered measurement. Hollow = held estimate (max 1s). Missing contacts stay unknown.',24,997,.43,MUTED)
        for name,(_,start,end) in specs.items():
            if start<=k<=end:writers[name].write(canvas)
        if k in [0,165,225,245,265,285,299]:cv2.imwrite(str(OUT/f'{"point_only_" if diagnostic else ""}comparison_{k:03}.jpg'),canvas)
        if k%100==0:print('Rendered frame',k,flush=True)
    cap.release()
    for w in writers.values():w.release()
    import imageio_ffmpeg
    for name in specs:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-i',str(OUT/(name+'_raw.mp4')),'-c:v','libx264','-crf','20','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/(name+'.mp4'))],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def report():
    scores=read(OUT/'accuracy.json');test=scores['development_test'];a=test['box_bottom'];b=test['learned_visible'];training=read(OUT/'training.json');cfg=read(OUT/'selection.json');counts=read(OUT/'label_counts.json');provenance=read(OUT/'provenance.json');stat=read(OUT/'stationary.json')
    improved=b['map']['floor_correct_within_3pct']>a['map']['floor_correct_within_3pct']
    verdict='More correct map positions on this small sample.' if improved else 'Point placement improves, but visibility rejection makes the complete map worse.'
    decision='Keep this as a candidate, then verify it on a fresh clip before expanding annotation.' if improved else 'Keep the support-point target as a candidate, but do not switch pipelines or start bulk labeling yet.'
    def table(split):
        rows=''
        names={'box_bottom':'Box bottom','learned_visible':'Trained support + visibility','learned_all':'Trained support, visibility disabled','box_learned_visibility':'Box bottom + learned visibility'}
        for name,d in scores[split].items():
            raw=d['raw'];m=d['map'];rows+=f"<tr><td>{names[name]}</td><td>{raw['correct_within_3pct']} / {raw['visible_references']}</td><td>{m['floor_correct_within_3pct']} / {raw['visible_references']}</td><td>{raw['median_error_pct_width']:.2f}%</td><td>{raw['anchors_emitted_on_hidden_supports']} / {raw['matched_hidden_supports']}</td></tr>"
        return '<table><tr><th>Method</th><th>Raw correct</th><th>Map correct</th><th>Median raw error</th><th>Hidden/unclear contacts reported</th></tr>'+rows+'</table>'
    definition='Mark the visible planted-foot center when standing. For kneeling, mark the supporting knee or center between supporting knees. A raised foot is not a support point. Leave the point absent when hidden, airborne or unclear; retain the dancer box and status. This is a chosen stance anchor, not a physical center of mass.'
    intro=f'We freshly annotated 21 frames: 12 for training, 4 for validation and 5 for evaluation. The training set has 240 dancer boxes, {counts["train"]["visible"]} visible support points and {240-counts["train"]["visible"]} hidden/uncertain contacts. Hidden contacts have no invented coordinate.'
    limitation='These are approximate assistant visual annotations, not independently reviewed ground truth. The test frames were excluded from this training and threshold selection, but the same clip has been used in previous pilots. This is development evidence, not a fresh generalization benchmark. Calibration and existing ID errors remain.'
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Trained floor-support pilot</title><style>
body{{background:#10191e;color:#e9f0f4;font:17px/1.65 system-ui;margin:0}}main{{max-width:1160px;margin:auto;padding:44px 24px 72px}}h1{{font-size:46px;line-height:1.12;max-width:900px}}h2{{font-size:25px;margin-top:36px}}.tag{{color:#98d5be;font-size:13px;letter-spacing:2px;text-transform:uppercase}}.card{{background:#1d2a31;padding:24px;border-radius:16px;margin:22px 0}}video,img{{width:100%;border-radius:10px}}a{{color:#a4d3ff}}button{{background:#294653;color:white;padding:12px 18px;border:0;border-radius:8px;margin:0 8px 15px 0;cursor:pointer}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #40505a}}th{{color:#a3dfc8}}.muted{{color:#acbec8}}.gallery{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}p{{max-width:1000px}}</style><main>
<div class="tag">Dance project · trained support-point pilot</div><h1>Label the floor contact.<br>Then test the whole map.</h1><p><strong>{verdict}</strong></p><p>{intro}</p>
<div class="card"><button onclick="play('support_comparison.mp4')">Support + visibility</button><button onclick="play('test_half_speed.mp4')">Test section · half speed</button><button onclick="play('point_only_comparison.mp4')">Diagnostic · visibility off</button><button onclick="play('point_only_test_half_speed.mp4')">Diagnostic test · half speed</button><video id="video" src="support_comparison.mp4" poster="comparison_285.jpg" controls playsinline preload="metadata"></video><p class="muted">Left: current box-bottom method. Right: the newly trained model. Both retain the same YOLO boxes, IDs and map filter. Hollow dots are held estimates. The diagnostic disables visibility rejection to isolate point placement; it can emit incorrect points on hidden supports.</p></div>
<h2>1. Label one consistent target</h2><p>{definition}</p><img src="audit/285.jpg" alt="Fresh visible support labels on the kneeling test frame"><p class="muted">White/yellow crosses are reference points. V = visible, H = hidden, U = uncertain. These are hand-picked visual estimates; small or blurred contacts need teammate review.</p><p><a href="LABEL_REVIEW.html">Review all 21 annotated frames</a> · <a href="CVAT_images.zip">Images for CVAT</a> · <a href="annotations.xml">CVAT annotations XML</a> · <a href="CVAT_SETUP.txt">Import instructions</a></p>
<h2>2. Train a small support model</h2><p>YOLO continues to detect dancers. A separate pretrained ResNet18 receives a crop around each detected dancer and learns a floor-point heatmap plus a visible-enough-to-report score. Point loss applies only to visible contacts. Hidden/uncertain examples train the second output.</p><p>We trained for {training['epochs_run']} epochs in {training['seconds']:.0f} seconds on the GPU. Validation selected epoch {training['best_epoch']} and a reportability threshold of {cfg['threshold']}. Crop jitter, flips and color changes expand the training views; they do not create new independent labels.</p><img src="training_curve.png" alt="Training and validation curves">
<h2>3. Evaluate on later frames</h2>{table('development_test')}<p class="muted">Five later frames contain 75 visible support references. Correct means within 3% of relative mat width. Missing points count against the numerator. Median error uses only available matched estimates, so read it with coverage. The last column checks whether the method wrongly emits a coordinate on a matched hidden/unclear support.</p><p>The two diagnostic rows separate point localization from visibility rejection. The current map retains its original overlap heuristic; the trained map uses learned reportability. Both use median-3, bounded EMA and the same maximum one-second hold.</p>
<h2>4. What validation showed</h2>{table('validation')}<p class="muted">These four frames selected the checkpoint and threshold. They are not the final result.</p>
<h2>5. Why the full model fails</h2><p>The visibility head accepts all 23 matched standing-foot contacts, but only 9 of 42 matched kneeling contacts. Training contains 109 visible foot examples and 28 knee examples. Validation has no visible knee examples. This split is too narrow to select a reliable visibility rule for floor work.</p><p>With visibility rejection disabled, raw correct positions rise from 54 to 62 out of 75, and correct filtered map positions rise from 55 to 60. That is promising localization evidence, but it also emits points on hidden contacts. In a short stationary check over 13 shared dancers, point-only map fluctuation rises from 0.065% to 0.205% of mat width. Better placement has not yet produced a smoother overall result.</p>
<h2>6. Decision</h2><div class="card"><strong>{decision}</strong><p>The next small dataset should include visible and hidden support for standing, kneeling and transitions, across front and back rows, in both training and validation. Keep a separate later choreography section untouched for the next evaluation. Review the labels together before assigning teammate work.</p><p>{limitation}</p></div><p><a href="Support_Point_Pilot.ipynb">Concise executed notebook</a> · <a href="accuracy.json">Complete scores</a> · <a href="reviewed_box_diagnostic.json">Localization with reviewed boxes</a> · <a href="tolerance_sensitivity.json">Stricter/looser tolerances</a> · <a href="point_only_stationary.json">Short stationary jitter check</a> · <a href="provenance.json">Frozen-input verification</a></p>
<p class="muted">The source clip is 300 frames at 25 fps. Training labels are frames 0–132 sampled every 12 frames; validation 165, 175, 185, 195; evaluation 225, 245, 265, 285, 299. Unlabelled frames are used for replay only. <a href="https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html">Torchvision ResNet18 documentation</a>.</p></main><script>function play(src){{const v=document.getElementById('video');v.src=src;v.load();v.play()}}</script></html>'''
    (OUT/'START_HERE.html').write_text(body,encoding='utf-8')
    gallery='<html><meta charset="utf-8"><title>Support labels</title><style>body{font:17px system-ui;background:#10191e;color:white;margin:30px}img{width:100%;max-width:1500px}a{color:#a4d3ff}section{margin:30px 0}</style><h1>Fresh support labels</h1><p>'+definition+'</p><p>Assistant-reviewed estimates. Review before using these as a team template.</p>'
    for k in FRAMES:gallery+=f'<section><h2>Frame {k} · {"TRAIN" if k in TRAIN else "VALIDATION" if k in VAL else "TEST"}</h2><a href="images/{k:03}.png">Original image</a><br><img src="audit/{k:03}.jpg"></section>'
    (OUT/'LABEL_REVIEW.html').write_text(gallery,encoding='utf-8')
    (OUT/'CVAT_SETUP.txt').write_text('CVAT image task import\n\n1. Create an image task and upload CVAT_images.zip. Keep filename sorting (000.png ... 299.png).\n2. Create labels dancer (rectangle) and floor_support (points). Both have text attributes dancer_id, contact, support_type. The JSON label specification is also provided.\n3. Import annotations.xml using CVAT for images 1.1.\n4. Review each point against the original image. Hidden/uncertain contacts have a dancer box and attribute, but NO point. Fully hidden people exist only in labels.json because no visible box can be drawn.\n5. Images are sparse independent keyframes, not an interpolated tracking task. Dancer IDs are attributes/group IDs.\n\nThese are approximate assistant visual labels for a pilot, not a finalized teammate labeling guide.\n',encoding='utf-8')
    write('cvat_labels.json',[{'name':name,'type':typ,'attributes':[{'name':a,'mutable':True,'input_type':'text','default_value':'','values':[]} for a in ['dancer_id','contact','support_type']]} for name,typ in [('dancer','rectangle'),('floor_support','points')]])
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    hist=read(OUT/'training_history.json');fig,axes=plt.subplots(1,2,figsize=(11,3.4),layout='constrained')
    for key,label in [('train_loss','Train'),('validation_loss','Validation')]:axes[0].plot([r['epoch'] for r in hist],[r[key] for r in hist],label=label)
    axes[0].axvline(training['best_epoch'],color='gray',ls='--');axes[0].set(xlabel='Epoch',ylabel='Combined loss');axes[0].legend();axes[1].plot([r['epoch'] for r in hist],[r['validation_median_pixel_error'] for r in hist]);axes[1].set(xlabel='Epoch',ylabel='Validation median point error (pixels)');fig.savefig(OUT/'training_curve.png',dpi=160);plt.close(fig)
    import nbformat as nbf
    from nbclient import NotebookClient
    nb=nbf.v4.new_notebook();nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}
    nb.cells=[nbf.v4.new_markdown_cell('# Learning the floor-support point\n\n'+intro+'\n\nQuestion: does a model trained on actual support labels beat the box-bottom map?'),nbf.v4.new_code_cell("import json\nfrom pathlib import Path\nfrom IPython.display import display, Image, Video\nroot=Path.cwd()\nload=lambda n: json.loads((root/n).read_text())\nload('label_counts.json')"),nbf.v4.new_markdown_cell('## 1. Define the label\n\n'+definition+'\n\nThe sparse frames are split by time. No hidden points are interpolated or used for point loss.'),nbf.v4.new_code_cell("display(Image(filename=str(root/'audit/285.jpg')))"),nbf.v4.new_markdown_cell('## 2. Train the support head\n\nKeep YOLO detection and tracking fixed. Fine-tune a pretrained ResNet18 on person crops. One head locates support; the other learns whether support is reportable. Choose the checkpoint and threshold using validation only.'),nbf.v4.new_code_cell("display(Image(filename=str(root/'training_curve.png')))\nload('training.json')"),nbf.v4.new_markdown_cell('## 3. Replay the same clip\n\nChange the map input, retain the existing map filter. Hidden support may retain a hollow estimate for up to one second, then disappears. It is never silently replaced by a box bottom.'),nbf.v4.new_code_cell("display(Video(str(root/'support_comparison.mp4'),embed=False,width=1100))"),nbf.v4.new_markdown_cell('## 4. Compare accuracy and availability\n\nUse all 75 visible references in the later section as the denominator. Missing estimates count as misses. Also count emitted points on hidden/unclear contacts. Median error by itself can reward a method that discards difficult dancers.'),nbf.v4.new_code_cell("for name,d in load('accuracy.json')['development_test'].items():\n    r=d['raw']; m=d['map']\n    print(name, 'raw correct',r['correct_within_3pct'],'/75; map correct',m['floor_correct_within_3pct'],'/75; hidden outputs',r['anchors_emitted_on_hidden_supports'])\nload('reviewed_box_diagnostic.json')"),nbf.v4.new_markdown_cell('## 5. Decision\n\n'+verdict+' '+decision+'\n\n'+limitation+'\n\nThe review package includes images, CVAT XML, labels, trained weights and code snapshots. The notebook reads saved results. To rerun training/inference, use the existing DL project and scripts listed in REPRODUCE.txt.\n\n[Torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)')]
    nb.cells.insert(-1,nbf.v4.new_markdown_cell('### What failed\n\nThe point head alone improves raw correct locations from 54 to 62 / 75. But visibility accepts only 9 of 42 matched knee contacts, versus all 23 matched foot contacts. Training contains 28 visible knee examples and validation contains none. The full system therefore loses too many dancers.\n\nDisabling visibility improves filtered correct positions to 60 / 75 versus 55 / 75, but permits hidden-contact guesses. It also increases short stationary map fluctuation from 0.065% to 0.205% of mat width over 13 paired dancers. We have evidence for better localization, not a complete stable solution.\n\nNext: balance visible/hidden support across standing, kneeling, transitions and stage rows in training and validation. Keep a fresh choreography section for evaluation before scaling teammate labeling.'))
    NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(OUT)}}).execute();nbf.write(nb,OUT/'Support_Point_Pilot.ipynb')
    (OUT/'REPRODUCE.txt').write_text('Run from the existing DL project root using Football YOLO trial/.venv/Scripts/python.exe:\n\nDance_Floor_Contact/support_labels.py\nDance_Floor_Contact/support_train.py\nDance_Floor_Contact/support_evaluate.py\nDance_Floor_Contact/support_deliver.py\n\nDependencies: existing Dance_Controlled_Test/source.mp4, Dance_Tracking_Pilot/tracks.json, Dance_CVAT_Pilot/calibration_draft.json, and helper modules in Dance_Floor_Contact. The review ZIP is portable for viewing, CVAT import and notebook execution; training reproduction uses the existing project layout.\n',encoding='utf-8')
    code=OUT/'code';code.mkdir(exist_ok=True)
    for name in ['support_label_setup.py','support_labels.py','support_train.py','support_evaluate.py','support_deliver.py']:shutil.copy2(Path(__file__).parent/name,code/name)
    import importlib.metadata as meta
    write('environment.json',{'python':sys.version,'packages':{n:meta.version(n) for n in ['torch','torchvision','ultralytics','opencv-python','numpy']}})

def verify(data):
    assert frozen_hash(data['current'])==frozen_hash(data['learned'])
    for row in data['learned']['frames']:
        assert len(row)==20
        for o in row:
            if o['map_status']=='estimated':assert 0<o['map_age_frames']<=25
            if o['map_status']=='unknown':assert o['map'] is None
    results={'same_6000_boxes_ids_status_records':True}
    for name,n,fps in [('support_comparison',300,25),('test_half_speed',75,12.5),('point_only_comparison',300,25),('point_only_test_half_speed',75,12.5)]:
        cap=cv2.VideoCapture(str(OUT/(name+'.mp4')));actual=cap.get(cv2.CAP_PROP_FPS);count=0
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape==(1000,1600,3);count+=1
        cap.release();assert count==n and actual==fps;results[name]={'frames':count,'fps':actual}
    write('verification.json',results)
    with zipfile.ZipFile(OUT/'Support_Pilot_Review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if not p.is_file() or p.suffix=='.zip' or p.name.endswith('_raw.mp4') or any(s in p.parts for s in ['review','person_review']):continue
            z.write(p,p.relative_to(OUT))
        z.write(OUT/'CVAT_images.zip','CVAT_images.zip')
    print(json.dumps(results),flush=True)

if __name__=='__main__':
    data=read(OUT/'tracks.json');labels=read(OUT/'labels.json');render(data,labels);render(data,labels,True);report();verify(data)
