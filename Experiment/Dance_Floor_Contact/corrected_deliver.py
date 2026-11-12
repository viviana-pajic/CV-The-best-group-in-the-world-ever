"""Render held-out and full-clip replays, plus a concise portable report."""
import json,shutil,subprocess,base64,html,colorsys
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import cv2,numpy as np
from corrected_setup import OUT,SOURCE,TRAIN,VAL,TEST,read
from render_controlled import text,color as seed_color,grid,INK,MUTED,BG
from compare import stage

NAMES={'baseline':'PRETRAINED YOLO / box-bottom map','tuned_box':'TUNED YOLO / box-bottom map','pose':'TUNED YOLO / learned support point','hybrid':'TUNED DETECTOR + learned support point'}
def color(i):
    if i<=20:return seed_color(i)
    return tuple(round(v*255) for v in colorsys.hsv_to_rgb(((i-100)*.61803398875)%1,.60,1)[::-1])
def draw_source(im,row):
    im=im.copy();grid(im)
    for o in row:
        i=o['id'];a,b,c,d=map(int,o['box']);co=color(i)
        cv2.rectangle(im,(a,b),(c,d),co,1,cv2.LINE_AA)
        label=str(i) if i<=20 else 'N'+str(i-100)
        if 'point' not in o:label+='?'
        cv2.rectangle(im,(a,b-14),(a+len(label)*8+5,b),BG,-1);text(im,label,a+2,b-3,.34,co)
        if 'point' in o:cv2.circle(im,tuple(np.rint(o['point']).astype(int)),3,co,-1)
    return im
def draw_map(canvas,x,y,series,k,refs=None):
    w,h=650,406
    cv2.rectangle(canvas,(x,y),(x+w,y+h),(35,33,30),-1)
    for u in np.linspace(0,1,9):cv2.line(canvas,(x+round(u*w),y),(x+round(u*w),y+h),(65,63,58),1)
    for v in np.linspace(0,1,7):cv2.line(canvas,(x,y+round(v*h)),(x+w,y+round(v*h)),(65,63,58),1)
    def loc(p):return round(x+p[0]/100*w),round(y+p[1]/62.5*h)
    def inside(p):return 0<=p[0]<=100 and 0<=p[1]<=62.5
    text(canvas,'BACK',x+300,y-8,.4,MUTED);text(canvas,'FRONT',x+300,y+h+20,.4,MUTED)
    if refs:
        for o in refs:
            if o['floor']:
                p=stage(o['floor'])
                if inside(p):cv2.drawMarker(canvas,loc(p),(240,240,240),cv2.MARKER_CROSS,13,1,cv2.LINE_AA)
    count=filled=0
    for ident,obj in series.items():
        i=int(ident);seq=obj['smooth'];key=str(k)
        if key not in seq or not inside(seq[key]):continue
        c=color(i);count+=1;gap=k in obj['filled'];filled+=gap
        for t in range(max(0,k-15),k):
            a,b=str(t),str(t+1)
            if a in seq and b in seq and inside(seq[a]) and inside(seq[b]):cv2.line(canvas,loc(seq[a]),loc(seq[b]),tuple(int(v*.5) for v in c),1,cv2.LINE_AA)
        if key in obj['raw'] and inside(obj['raw'][key]):cv2.circle(canvas,loc(obj['raw'][key]),2,(150,150,150),-1)
        p=loc(seq[key]);cv2.circle(canvas,p,6,c,2 if gap else -1,cv2.LINE_AA)
        text(canvas,str(i) if i<=20 else 'N'+str(i-100),p[0]+8,p[1]+3,.36,c)
    return count,filled
def render(mode_override=None):
    settings=read(OUT/'operating_points.json');selected=mode_override or settings['selected_pipeline'];tracks=read(OUT/'tracks.json');labels=read(OUT/'labels.json')
    prefix='detector_' if mode_override else ''
    cap=cv2.VideoCapture(str(SOURCE/'source.mp4'))
    writers={prefix+name:cv2.VideoWriter(str(OUT/(prefix+name+'_raw.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1600,1000)) for name,fps in [('full_comparison',25),('test_comparison',12.5)]}
    for k in range(300):
        ok,im=cap.read();assert ok
        canvas=np.full((1000,1600,3),BG,np.uint8)
        phase='TRAINING PERIOD' if k<=143 else 'VALIDATION PERIOD' if 165<=k<=195 else 'HELD-OUT TEST' if k>=225 else 'UNUSED GAP'
        text(canvas,'CORRECTED LABELS / BEFORE AND AFTER TRAINING',24,36,.75,INK,2)
        text(canvas,f'{phase}   |   clip +{k/25:05.2f}s   |   same calibration, crops and tracker',24,65,.48,(150,230,185) if k>=225 else MUTED)
        refs=labels.get(str(k))
        for col,mode in enumerate(('baseline',selected)):
            x=24+800*col;data=tracks[mode];row=data['frames'][k]
            text(canvas,NAMES[mode],x,101,.58,INK,2)
            text(canvas,f'{len(row)} active tracks  |  {sum("point" in o for o in row)} support estimates before smoothing',x,128,.45,MUTED)
            annotated=draw_source(im,row)
            canvas[145:407,x:x+752]=cv2.resize(annotated[285:635,100:1200],(752,262))
            text(canvas,'STAGE MAP / identical seven-frame smoothing',x,447,.47,INK)
            dots,gaps=draw_map(canvas,x+51,481,data['series'],k,refs)
            text(canvas,f'{dots} map dots  |  {gaps} short-gap estimates',x+51,938,.46,MUTED)
        text(canvas,'White crosses: reviewed references (on labeled frames)  |  Hollow dots: interpolation  |  ?: no accepted support point',24,970,.43,MUTED)
        text(canvas,'Relative mat coordinates; draft calibration. Test-only replay is slowed to 0.5x. No later manual corrections to predictions.',24,992,.41,MUTED)
        writers[prefix+'full_comparison'].write(canvas)
        if k>=225:writers[prefix+'test_comparison'].write(canvas)
        if k in [0,138,195,*TEST]:cv2.imwrite(str(OUT/f'{prefix}comparison_{k:03}.jpg'),canvas)
    cap.release()
    for w in writers.values():w.release()
    import imageio_ffmpeg
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    for name in writers:
        subprocess.run([ffmpeg,'-y','-i',str(OUT/(name+'_raw.mp4')),'-c:v','libx264','-crf','20','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/(name+'.mp4'))],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    print('Rendered',selected,flush=True)
def page():
    metrics=read(OUT/'test_metrics.json');selected=metrics['selected_using_validation'];settings=read(OUT/'operating_points.json');summary=read(OUT/'dataset_summary.json');tracking=read(OUT/'tracking_metrics.json')
    a,b=metrics['baseline'],metrics['tuned_box'];learned=metrics['pose']
    rows=''
    def pct(v):return f'{v*100:.1f}%'
    fields=[('Matched visible dancers',lambda s:f"{s['boxes']['matched_iou50']} / {s['boxes']['ground_truth']}"),('Detection precision',lambda s:pct(s['boxes']['precision'])),('Detection recall',lambda s:pct(s['boxes']['recall'])),('Correct floor positions / all visible references',lambda s:f"{s['floor']['correct_within_3pct_mat_width']} / {s['floor']['visible_references']}"),('Floor-point precision (within tolerance)',lambda s:pct(s['floor']['precision_within_3pct'])),('Median floor error on matched outputs',lambda s:f"{s['floor']['median_error_pct_mat_width']:.2f}% of mat width"),('Points emitted for hidden/uncertain supports',lambda s:str(s['floor']['points_emitted_on_matched_hidden_supports']))]
    for title,fn in fields:rows+=f'<tr><td>{title}</td><td>{fn(a)}</td><td>{fn(b)}</td><td>{fn(learned)}</td></tr>'
    improvement=b['floor']['correct_fraction_of_all_visible_references']-a['floor']['correct_fraction_of_all_visible_references']
    verdict=f"Detector-only fine-tuning gave a small improvement: correct floor proxies changed from {a['floor']['correct_within_3pct_mat_width']}/{a['floor']['visible_references']} to {b['floor']['correct_within_3pct_mat_width']}/{b['floor']['visible_references']}, and detection recall from {pct(a['boxes']['recall'])} to {pct(b['boxes']['recall'])}. The learned support-point model fell to {learned['floor']['correct_within_3pct_mat_width']}/{learned['floor']['visible_references']}."
    review=''.join(f'<a href="detector_comparison_{k:03}.jpg">+{k/25:.2f}s</a> ' for k in TEST)
    fixed=read(OUT/'fixed_threshold_metrics.json');fa,fb=fixed['baseline'],fixed['tuned_box']
    fixed_note=f"At the same detector confidence of 0.25, recall changes from {pct(fa['boxes']['recall'])} to {pct(fb['boxes']['recall'])}, while precision changes from {pct(fa['boxes']['precision'])} to {pct(fb['boxes']['precision'])}. The extra detections therefore include extra false positives."
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dance · corrected-label training pilot</title><style>*{{box-sizing:border-box}}body{{background:#10141c;color:#ecf2fa;font:16px/1.6 system-ui;margin:0}}main{{max-width:1320px;margin:auto;padding:32px 24px}}h1{{font-size:clamp(28px,4vw,46px);line-height:1.15}}h2{{font-size:23px}}p{{max-width:950px;color:#bac8da}}.tag{{font-size:12px;letter-spacing:.13em;color:#80dfbd;font-weight:700}}video{{width:100%;border-radius:14px;border:1px solid #3c4859;background:black}}button,a.button{{display:inline-block;background:#263248;color:#edf2fa;border:1px solid #4c5a70;border-radius:7px;padding:9px 15px;cursor:pointer;margin:8px 8px 8px 0;font:inherit;text-decoration:none}}.card{{background:#192230;padding:24px;border:1px solid #354257;border-radius:14px;margin:24px 0}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:11px;border-bottom:1px solid #354257}}a{{color:#91e6d0}}img{{max-width:100%;border-radius:12px}}small{{color:#b3c1d4}}code{{background:#263248;padding:2px 5px}}</style>
    <main><div class="tag">DANCE FORMATIONS / TRAINING PILOT</div><h1>Do corrected labels help?</h1><p>{verdict}</p>
    <video id="v" controls loop playsinline preload="metadata" poster="detector_comparison_285.jpg"><source src="detector_test_comparison.mp4" type="video/mp4"></video>
    <div><button onclick="let v=document.getElementById('v');v.src='detector_test_comparison.mp4';v.play()">Detector-only test · 0.5×</button><button onclick="let v=document.getElementById('v');v.src='test_comparison.mp4';v.play()">Learned-point test · 0.5×</button><button onclick="let v=document.getElementById('v');v.src='detector_full_comparison.mp4';v.play()">Detector full clip</button><button onclick="let v=document.getElementById('v');v.src='full_comparison.mp4';v.play()">Learned-point full clip</button><a class="button" href="detector_test_comparison.mp4" download>Download detector test</a></div>
    <p>The main replay contains only the final three seconds, which were excluded from training and model selection. The full replay labels the training, validation and test periods.</p>
    <section class="card"><h2>Test results</h2><p>Eight reviewed frames. One-to-one box matching at IoU 0.5. A correct support point must be within 3% of the displayed mat width.</p><table><thead><tr><th>Metric</th><th>Pretrained baseline</th><th>Tuned detector only</th><th>Learned support model</th></tr></thead><tbody>{rows}</tbody></table><p>Validation selected the learned support model, but it performed worse on the later kneeling test. The detector-only result is shown separately; we have not changed the recorded model selection. Missing predictions remain in the denominator.</p><p>Between reviewed test frames, the baseline had {tracking['baseline']['id_changes_between_reviewed_frames']} ID changes across {tracking['baseline']['reviewed_identity_pairs']} matched pairs; the tuned detector had {tracking['tuned_box']['id_changes_between_reviewed_frames']} across {tracking['tuned_box']['reviewed_identity_pairs']}. These are sampled checks, not a full identity benchmark.</p></section>
    <h2>What was corrected</h2><p>36 selected keyframes: 24 training, 4 validation, 8 test. Every identity was reviewed. The training set contains {summary['train']['visible_boxes']} boxes and {summary['train']['support_points']} support points. Completely hidden people have no detection box. Hidden or uncertain supports have no guessed point. The other 264 frames are used for replay, not as fabricated training examples.</p>
    <h2>What was trained</h2><p>A pretrained YOLO26m detector and a YOLO26m model with one custom support keypoint. Both use three overlapping crops at an input size of 960. The support-point model also learns when no reliable point is available. Its keypoint tolerance is explicitly set to 0.1 instead of the default 1.0 for a one-point model.</p>
    <h2>Read the video</h2><p>The calibration and ByteTrack parameters are identical for every model. White map crosses show the reviewed references on labeled frames. A question mark means that a tracked person has no accepted support estimate. Hollow map dots fill short gaps of up to four frames. Initial IDs 1–20 are shared; new N IDs are independent in each method.</p>
    <p>Review test stills: {review}</p>
    <h2>Check the tradeoff</h2><p>The score table uses confidence levels selected on validation. {fixed_note} The replay uses identical tracker settings on both sides. <a href="fixed_threshold_metrics.json">Fixed-confidence results</a>.</p><h2>Decision</h2><p>Keep the detector checkpoint for further testing: it retains more tracks, but the accuracy gain is small and includes a precision/recall tradeoff. The learned support model is not ready to replace the box-bottom baseline. This pilot does not justify assigning a large segmentation-labeling workload. Before scaling, refine the support-point policy on a few difficult examples and verify a new section; accurate tracking through complete occlusion remains unresolved.</p><h2>Limits</h2><p>These are assistant-reviewed, approximate annotations from one video, with temporal gaps between splits. This is a small pilot, not an independent human audit or a cross-video benchmark. Fully occluded people still need temporal reasoning. The mat calibration is unchanged and coordinates are relative, not metres.</p>
    <p><a href="test_metrics.json">All test results</a> · <a href="tracking_metrics.json">Tracking diagnostics</a> · <a href="operating_points.json">Frozen validation choices</a> · <a href="Corrected_Label_Pilot.ipynb">Concise notebook</a> · <a href="corrected_annotations.zip">Corrected labels and images</a></p>
    <details><summary>Inspect corrected starting labels</summary><img src="audit/000.jpg" alt="Reviewed boxes and visible support points"></details><details><summary>Inspect corrected final labels</summary><img src="audit/299.jpg" alt="Reviewed final-frame labels"></details></main></html>'''
    (OUT/'START_HERE.html').write_text(page,encoding='utf-8')
    with ZipFile(OUT/'corrected_annotations.zip','w',ZIP_DEFLATED) as z:
        for f in ('corrected_keyframes_cvat.xml','labels.json','manifest.json','dataset_summary.json','protocol.json','visual_edits.json'):z.write(OUT/f,f)
        for f in (OUT/'images').glob('*.png'):z.write(f,'images/'+f.name)
    (OUT/'result_summary.json').write_text(json.dumps({'selected':selected,'verdict':verdict,'floor_coverage_change_percentage_points':100*improvement},indent=2))
    print(verdict,flush=True)
if __name__=='__main__':render();render('tuned_box');page()
