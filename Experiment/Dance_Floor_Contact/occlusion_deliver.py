"""Portable video/report and paired checks for the tracking pilot."""
import json, subprocess, shutil, html
from pathlib import Path
from collections import Counter
import cv2, numpy as np
from occlusion_pilot import OUT, PREV, SOURCE, read, save, inside
from corrected_setup import VAL,TEST
from corrected_deliver import color
from compare import stage, match_boxes
from render_controlled import text,grid,BG,INK,MUTED

def checks(data,labels):
    residuals={m:{} for m in ['bytetrack','roster']}
    start={o['id']:o for o in labels['285']};end={o['id']:o for o in labels['299']}
    for k in range(285,300):
        a=(k-285)/14
        refs=[{'id':i,'box':np.array(o['box'])*(1-a)+np.array(end[i]['box'])*a,'floor':np.array(o['floor'])*(1-a)+np.array(end[i]['floor'])*a} for i,o in start.items() if o['floor'] is not None and end[i]['floor'] is not None and o['box'] is not None and end[i]['box'] is not None]
        for mode in residuals:
            rows=[o for o in data[mode]['frames'][k] if o.get('status','observed')=='observed' and o.get('map_status')=='proxy' and o.get('map') is not None]
            links=match_boxes([o['box'] for o in refs],[o['box'] for o in rows],.5)
            for i,j in links.items():residuals[mode].setdefault(refs[i]['id'],{})[k]=np.array(rows[j]['map'])-stage(refs[i]['floor'])
    paired={i:sorted(set(residuals['bytetrack'].get(i,{}))&set(residuals['roster'].get(i,{}))) for i in start}
    paired={i:kk for i,kk in paired.items() if len(kk)>=12}
    stat={'scope':'Approximate final 0.56-second stationary check. Interpolated reviewed endpoints used only for evaluation. Paired by spatial box match, not assumed tracker ID. Only solid observed proxies, never predicted/held dots. Not a full-clip jitter benchmark.','paired_reference_ids':list(paired),'paired_frames_per_id':{i:len(kk) for i,kk in paired.items()}}
    for mode in residuals:
        vals={}
        for i,kk in paired.items():
            arr=np.array([residuals[mode][i][k] for k in kk]);vals[i]=float(np.sqrt(np.mean(np.sum((arr-arr.mean(axis=0))**2,axis=1))))
        stat[mode]={'rms_by_id_pct_mat_width':vals,'median_rms_pct_mat_width':float(np.median(list(vals.values()))) if vals else None}
    save('paired_stationary_check.json',stat)
    summaries={}
    for mode in ('bytetrack','ocsort','roster'):
        frames=data[mode]['frames'];stats=[]
        for row in frames:
            states=Counter(o.get('status','observed') for o in row);maps=Counter(o.get('map_status','unknown') for o in row)
            stats.append({**{s:states[s] for s in ('observed','predicted','unknown')},**{s+'_map':maps[s] for s in ('proxy','estimated','unknown')}})
        summaries[mode]={key:{'mean':float(np.mean([s[key] for s in stats])),'minimum':min(s[key] for s in stats),'maximum':max(s[key] for s in stats)} for key in stats[0]}
    save('coverage.json',summaries)
    # Hard invariants for honesty of the display and roster.
    for row in data['roster']['frames']:
        assert len(row)==20 and len({o['id'] for o in row})==20
        for o in row:
            if o['status']=='predicted':assert 0<o['age_frames']<=25
            if o['map_status']=='estimated':assert 0<o['map_age_frames']<=25
            if o['map_status']=='unknown':assert o['map'] is None
    return stat,summaries

def dashed(im,a,b,c,thickness=1):
    a=np.array(a,float);b=np.array(b,float);dist=np.linalg.norm(b-a)
    for d in np.arange(0,dist,10):
        p=a+(b-a)*d/max(dist,1);q=a+(b-a)*min(d+5,dist)/max(dist,1)
        cv2.line(im,tuple(p.astype(int)),tuple(q.astype(int)),c,thickness,cv2.LINE_AA)

def source_frame(im,rows,extras=()):
    im=im.copy();grid(im)
    for b in extras:
        x1,y1,x2,y2=map(round,b);c=(140,140,140)
        for a,z in [((x1,y1),(x2,y1)),((x2,y1),(x2,y2)),((x2,y2),(x1,y2)),((x1,y2),(x1,y1))]:dashed(im,a,z,c)
        text(im,'?',x1,y1-3,.34,c)
    for o in rows:
        if o.get('status')=='unknown':continue
        b=list(map(round,o['box']));a,y,c,z=b;co=color(o['id']);pred=o.get('status')=='predicted'
        if pred:
            for p,q in [((a,y),(c,y)),((c,y),(c,z)),((c,z),(a,z)),((a,z),(a,y))]:dashed(im,p,q,co)
        else:cv2.rectangle(im,(a,y),(c,z),co,1,cv2.LINE_AA)
        label=('~' if pred else '')+(str(o['id']) if o['id']<=20 else 'N'+str(o['id']-100))
        cv2.rectangle(im,(a,y-14),(a+len(label)*8+4,y),BG,-1);text(im,label,a+2,y-3,.34,co)
    return im

def draw_map(im,x,y,frames,k,refs):
    w,h=650,406
    cv2.rectangle(im,(x,y),(x+w,y+h),(35,33,30),-1)
    for u in np.linspace(0,1,9):cv2.line(im,(x+round(u*w),y),(x+round(u*w),y+h),(65,63,58),1)
    for v in np.linspace(0,1,7):cv2.line(im,(x,y+round(v*h)),(x+w,y+round(v*h)),(65,63,58),1)
    def loc(p):return round(x+p[0]/100*w),round(y+p[1]/62.5*h)
    text(im,'BACK',x+300,y-8,.4,MUTED);text(im,'FRONT',x+300,y+h+20,.4,MUTED)
    for o in refs:
        if o['floor'] is not None:
            p=stage(o['floor'])
            if inside(p):cv2.drawMarker(im,loc(p),(240,240,240),cv2.MARKER_CROSS,13,1,cv2.LINE_AA)
    count=Counter()
    for o in frames[k]:
        p=o.get('map')
        if not inside(p):continue
        ident=o['id'];co=color(ident);estimated=o.get('map_status')=='estimated';count['estimated' if estimated else 'proxy']+=1
        previous=None
        for f in range(max(0,k-15),k+1):
            row=next((r for r in frames[f] if r['id']==ident and inside(r.get('map'))),None)
            if row:
                pp=loc(row['map'])
                if previous:cv2.line(im,previous,pp,tuple(round(c*.4) for c in co),1,cv2.LINE_AA)
                previous=pp
            else:previous=None
        point=loc(p)
        if estimated:cv2.circle(im,point,max(8,round(o.get('uncertainty',1)*6.5)),tuple(round(c*.45) for c in co),1,cv2.LINE_AA)
        cv2.circle(im,point,6,co,2 if estimated else -1,cv2.LINE_AA)
        label=('~' if estimated else '')+(str(ident) if ident<=20 else 'N'+str(ident-100))
        text(im,label,point[0]+8,point[1]+3,.36,co)
    return count

def render(data,labels):
    cap=cv2.VideoCapture(str(SOURCE/'source.mp4'));writers={name:cv2.VideoWriter(str(OUT/(name+'_raw.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1600,1000)) for name,fps in [('tracking_comparison',25),('overlap_slow_motion',12.5)]}
    for k in range(300):
        ok,im=cap.read();assert ok
        canvas=np.full((1000,1600,3),BG,np.uint8)
        text(canvas,'KEEPING IDENTITIES THROUGH OVERLAP',24,36,.80,INK,2)
        text(canvas,f'Development replay  |  +{k/25:05.2f}s  |  same tuned YOLO detections and first-frame boxes',24,65,.49,MUTED)
        for col,mode in enumerate(('bytetrack','roster')):
            x=24+800*col;row=data[mode]['frames'][k];states=Counter(o.get('status','observed') for o in row)
            text(canvas,'BEFORE / ByteTrack' if col==0 else 'AFTER / stage + body association',x,101,.62,INK,2)
            text(canvas,f"Detected: {states['observed']}   Predicted: {states['predicted']}   Unknown: {states['unknown']}" if col else f"{len(row)} active tracks; new IDs appear as N1, N2...",x,128,.45,MUTED)
            extras=data[mode]['diagnostics'][k]['unassigned_high_confidence_boxes'] if col else []
            annotated=source_frame(im,row,extras)
            # Preserve aspect ratio and the full image width, including both mat edges.
            canvas[160:395,x:x+752]=cv2.resize(annotated[275:675,:],(752,235))
            text(canvas,'STAGE MAP / box-bottom proxies' if col==0 else 'STAGE MAP / filtered proxies + bounded estimates',x,447,.47,INK)
            cnt=draw_map(canvas,x+51,481,data[mode]['frames'],k,labels.get(str(k),[]))
            text(canvas,f"{cnt['proxy']} solid proxies   |   {cnt['estimated']} hollow estimates",x+51,932,.45,MUTED)
            if col:
                missing=[str(o['id']) for o in row if o['status']=='unknown']
                text(canvas,'Unknown identities: '+(', '.join(missing) or 'none'),x+51,954,.40,MUTED)
        text(canvas,'Solid = detected person / floor proxy. Hollow or dashed ~ = prediction, max 1s. Gray ? = unassigned candidate.',24,976,.42,MUTED)
        text(canvas,'White crosses = sparse reviewed floor references. Visibility flags are heuristic. Relative mat calibration unchanged.',24,996,.41,MUTED)
        writers['tracking_comparison'].write(canvas)
        if k>=225:writers['overlap_slow_motion'].write(canvas)
        if k in [0,50,100,165,195,225,255,285,299]:cv2.imwrite(str(OUT/f'comparison_{k:03}.jpg'),canvas)
    cap.release()
    for writer in writers.values():writer.release()
    import imageio_ffmpeg
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    for name in writers:
        subprocess.run([ffmpeg,'-y','-i',str(OUT/(name+'_raw.mp4')),'-c:v','libx264','-crf','20','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/(name+'.mp4'))],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    print('Rendered comparison videos',flush=True)

def report(stat,coverage):
    metrics=read(OUT/'metrics.json');m=metrics['development_test'];a,b=m['bytetrack'],m['roster']
    rows=''
    for name,key in [('Visible dancer boxes matched / 145','matched_boxes'),('Matches retaining the original dancer ID / 145','correct_initial_ids'),('Matched boxes with a wrong or replacement ID','wrong_initial_ids'),('Correct floor positions, irrespective of ID / 120','floor_correct_within_3pct'),('Correct floor position AND original ID / 120','floor_and_initial_id_correct'),('Fully hidden IDs incorrectly claimed observed / 15','fully_hidden_ids_claimed_observed')]:
        rows+=f'<tr><td>{name}</td><td>{a[key]}</td><td>{b[key]}</td></tr>'
    fluct='Not enough common observations for the stationary check.'
    if stat['bytetrack']['median_rms_pct_mat_width'] is not None:
        aa=stat['bytetrack']['median_rms_pct_mat_width'];bb=stat['roster']['median_rms_pct_mat_width']
        fluct=f"The short stationary check covered {len(stat['paired_reference_ids'])} common dancers during the final 0.56 seconds. Median residual fluctuation was {aa:.3f}% → {bb:.3f}% of mat width. Only observed solid proxies were included. This is not a full-video jitter measurement."
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dance tracking pilot</title><style>
body{{background:#101517;color:#e9eef0;font:17px/1.6 system-ui;margin:0}}main{{max-width:1120px;margin:auto;padding:44px 24px 70px}}h1{{font-size:44px;line-height:1.12;max-width:800px}}h2{{font-size:25px;margin-top:36px}}p{{max-width:900px}}.eyebrow{{color:#85d5b2;text-transform:uppercase;letter-spacing:2px;font-size:13px}}.card{{background:#1c2529;padding:22px;border-radius:14px;margin:24px 0}}video{{width:100%;background:#060808;border-radius:10px}}button,a{{color:#91d7ff}}button{{background:#29383e;border:1px solid #4b626b;padding:12px 18px;border-radius:8px;cursor:pointer;margin:0 8px 12px 0}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #354449}}th{{color:#85d5b2}}.muted{{color:#a8b8bf}}code{{color:#d6dfad}}li{{margin-bottom:7px}}</style><main>
<div class="eyebrow">Dance project · tracking experiment · 12-second clip</div><h1>Better continuity.<br>Hidden dancers still need caution.</h1>
<p>The same tuned detector now feeds a tracker that keeps the 20 starting identities, uses floor-proxy distance and body overlap, and marks short gaps as predictions. It substantially improves identity retention on this replay. It does not solve every occlusion or recover every dancer.</p>
<div class="card"><button onclick="setVideo('tracking_comparison.mp4')">Full 12-second comparison</button><button onclick="setVideo('overlap_slow_motion.mp4')">Final 3 seconds · half speed</button><video id="v" controls playsinline preload="metadata" poster="comparison_285.jpg" src="tracking_comparison.mp4"></video><p class="muted">Left: ByteTrack. Right: revised tracker. Solid dots are box-bottom floor proxies. Hollow dots and dashed boxes are predictions, shown for at most one second. Gray ? boxes are detections not assigned to a roster ID. The full mat grid stays visible.</p></div>
<h2>What changed</h2><ol><li>Keep the 20 IDs initialized on the first frame; avoid automatically inventing new dancer IDs.</li><li>Associate detections using stage-proxy proximity and body overlap. Rescue a changing pose with body overlap when its box bottom becomes unreliable.</li><li>Retain lost identities for this clip, with stricter gates for reacquisition after three seconds. Display their location for only one second; after that it is unknown.</li><li>Reject isolated bottom-point jumps, filter accepted points, and flag possible floor occlusion from overlapping bodies. Sustained movement is allowed through after three consistent measurements.</li></ol>
<h2>What the reviewed final section shows</h2><table><thead><tr><th>Measure</th><th>Before</th><th>After</th></tr></thead><tbody>{rows}</tbody></table>
<p class="muted">Counts aggregate eight reviewed frames, not eight different groups of people. A box matches at IoU ≥ 0.5. Floor tolerance is 3% of mat width. Original-ID scoring compares each matched prediction to the reviewed dancer identity. Predictions are excluded from observed-box scores. Floor scores can include explicitly estimated points on an observed person.</p>
<p>{fluct}</p>
<div class="card"><strong>Decision: keep developing the tracker; do not start bulk labeling yet.</strong><p>Correct ID-and-position matches improved from {a['floor_and_initial_id_correct']} to {b['floor_and_initial_id_correct']} out of 120 reviewed floor references. But visible-box matches dropped from {a['matched_boxes']} to {b['matched_boxes']}, and the tracker still assigns some fully hidden IDs to visible people. Better-looking motion is not proof of correct identity.</p></div>
<h2>What this cannot establish</h2><p>This is a development replay of the same clip. The last three seconds were excluded from detector training, but we have already reviewed them in earlier experiments. Six association settings were compared on four validation frames; results are exploratory. Labels are approximate assistant-reviewed references, not independently audited ground truth. No later labels or manually corrected floor points are fed into tracking.</p>
<p>All methods use the same cached tuned YOLO detections, crop windows, calibration, and 20 first-frame boxes. ByteTrack uses its existing seven-frame offline smoothing; the new filter is causal. OC-SORT was also tested with the same detector: it retained {m['ocsort']['correct_initial_ids']} original-ID matches here, so it was not an improvement.</p>
<p>Solid dots are still floor <em>proxies</em>, not verified foot contacts. The occlusion rule is a geometric warning, not a trained visibility model. The assumed mat aspect ratio and calibration are unchanged. The roster assumes the same 20 dancers; new entrances require separate handling.</p>
<h2>The next small change worth testing</h2><p>Review the actual identity switches at overlap boundaries, then add appearance-assisted reassignment or a manual correction at those few moments. For training data, preserve whole-person boxes, dancer IDs across short sequences, and visible versus unknown floor contacts. Avoid drawing guessed contacts for fully hidden dancers. This pilot does not justify labeling every frame or switching to full segmentation.</p>
<h2>Files and reproduction</h2><p><a href="Tracking_Pilot.ipynb">Concise notebook</a> · <a href="metrics.json">Scores</a> · <a href="settings.json">Selected settings</a> · <a href="selection.json">All six validation runs</a> · <a href="provenance.json">Provenance and limits</a></p>
<p class="muted">Reference methods: <a href="https://arxiv.org/abs/2203.14360">OC-SORT paper</a> and <a href="https://docs.ultralytics.com/modes/track/">Ultralytics tracking documentation</a>. The revised tracker is a custom experiment, not an implementation of OC-SORT.</p>
</main><script>function setVideo(s){{const v=document.getElementById('v');v.src=s;v.load();v.play();}}</script></html>'''
    (OUT/'START_HERE.html').write_text(body,encoding='utf-8')
    import nbformat as nbf
    nb=nbf.v4.new_notebook();nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}
    nb.cells=[nbf.v4.new_markdown_cell('# Tracking through overlap\n\nWe kept the trained detector fixed and changed how detections are joined over time. This is a development replay, not a fresh independent test.'),nbf.v4.new_code_cell("import json\nfrom pathlib import Path\nfrom IPython.display import Video, display\nroot = Path.cwd()\nmetrics = json.loads((root / 'metrics.json').read_text())\nsettings = json.loads((root / 'settings.json').read_text())\nsettings"),nbf.v4.new_markdown_cell('## 1. Keep identities, allow uncertainty\n\nAll methods start with the same 20 reviewed boxes. The revised method combines body overlap and stage-proxy distance. Unmatched identities remain in memory, but their predicted locations disappear after one second. Hidden floor contact is not treated as a measured point.'),nbf.v4.new_markdown_cell('## 2. Make box bottoms less jumpy\n\nReject isolated jumps, require three consistent readings for a large change, and smooth accepted observations. Mark possible occlusion when another body covers the bottom-center point. This is a heuristic; it cannot certify contact visibility.'),nbf.v4.new_code_cell("keys = ['matched_boxes', 'correct_initial_ids', 'wrong_initial_ids', 'floor_correct_within_3pct', 'floor_and_initial_id_correct', 'fully_hidden_ids_claimed_observed']\nfor key in keys:\n    print(f\"{key}: {metrics['development_test']['bytetrack'][key]} -> {metrics['development_test']['roster'][key]}\")"),nbf.v4.new_markdown_cell('## 3. Watch the result\n\nLeft: existing ByteTrack. Right: revised tracker. Solid dots are floor proxies; hollow dots are short predictions; unknown locations are omitted. The grid uses the existing relative mat calibration.'),nbf.v4.new_code_cell("display(Video(str(root / 'tracking_comparison.mp4'), embed=False, width=1000))"),nbf.v4.new_markdown_cell('## 4. Check stability without rewarding frozen guesses\n\nThe short stationary check pairs the same spatially matched dancers and frames. It excludes predicted dots. Endpoint interpolation is used only for evaluation, never to create tracking outputs.'),nbf.v4.new_code_cell("json.loads((root / 'paired_stationary_check.json').read_text())"),nbf.v4.new_markdown_cell(f"## 5. Decision\n\nCorrect original-ID matches improved from {a['correct_initial_ids']} to {b['correct_initial_ids']}, but visible-box matches fell from {a['matched_boxes']} to {b['matched_boxes']} out of 145. Hidden-person identity errors remain. Keep this as a tracking prototype; defer bulk labeling. The next check should focus on ID switches at overlap boundaries.\n\nLimitations: four validation frames, eight previously reviewed final-section frames, approximate labels, one clip, a fixed 20-person roster, and unchanged calibration. Six parameter combinations were considered; their validation scores are saved in `selection.json`. No retraining was performed for this experiment."),nbf.v4.new_markdown_cell('## Reproduce\n\nRun `occlusion_pilot.py`, then `occlusion_deliver.py` using the project Python environment from the DL project root. The scripts read the existing detector cache and first-frame boxes from the previous pilot. The `reproduce` folder includes the scripts and their local dependencies; the previous project folders and source clip are still required.\n\nReferences: [Ultralytics tracking](https://docs.ultralytics.com/modes/track/) and [OC-SORT](https://arxiv.org/abs/2203.14360).')]
    nbf.write(nb,OUT/'Tracking_Pilot.ipynb')
    from nbclient import NotebookClient
    NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(OUT)}}).execute()
    nbf.write(nb,OUT/'Tracking_Pilot.ipynb')
    scripts=OUT/'reproduce';scripts.mkdir(exist_ok=True)
    for name in ['occlusion_pilot.py','occlusion_deliver.py','compare.py','corrected_setup.py','corrected_deliver.py','render_controlled.py','infer.py']:
        shutil.copy2(Path(__file__).parent/name,scripts/name)
    (OUT/'REPRODUCE.txt').write_text('Run from the original DL project root using the existing Python environment:\n\n"Football YOLO trial/.venv/Scripts/python.exe" "Dance_Floor_Contact/occlusion_pilot.py"\n"Football YOLO trial/.venv/Scripts/python.exe" "Dance_Floor_Contact/occlusion_deliver.py"\n\nThe reproduce folder contains reference copies. The scripts expect to live in Dance_Floor_Contact alongside the existing project folders; do not run those copies in place. Required inputs include Dance_Corrected_Pilot, Dance_Controlled_Test/source.mp4, Dance_CVAT_Pilot/calibration_draft.json, and the existing supporting project modules. The HTML, videos, notebook outputs and JSON reports can be viewed without those inputs.\n',encoding='utf-8')
    print('Wrote report and executed notebook',flush=True)

def verify():
    results={}
    for name,expected,fps in [('tracking_comparison.mp4',300,25),('overlap_slow_motion.mp4',75,12.5)]:
        cap=cv2.VideoCapture(str(OUT/name));actual_fps=cap.get(cv2.CAP_PROP_FPS);count=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape==(1000,1600,3);count+=1
        cap.release();assert count==expected and abs(actual_fps-fps)<.01
        results[name]={'decoded_frames':count,'fps':actual_fps,'size_bytes':(OUT/name).stat().st_size}
    save('verification.json',results);print(json.dumps(results),flush=True)

if __name__=='__main__':
    data=read(OUT/'tracks.json');labels=read(PREV/'labels.json');stat,cov=checks(data,labels);render(data,labels);report(stat,cov);verify()
