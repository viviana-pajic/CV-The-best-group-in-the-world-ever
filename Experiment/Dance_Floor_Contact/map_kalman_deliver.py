"""Render the frozen-detection, map-only comparison and honest experiment report."""
import json,subprocess,shutil
from collections import Counter
from pathlib import Path
import cv2,numpy as np
from map_kalman_pilot import OUT,LABEL_DIR,SOURCE,read,write,frozen_hash,stage
from occlusion_deliver import source_frame,draw_map
from render_controlled import text,BG,INK,MUTED

def render(data,labels):
    specs={'map_comparison':(25,0,299),'movement_half_speed':(12.5,0,149),'overlap_half_speed':(12.5,225,299)}
    writers={n:cv2.VideoWriter(str(OUT/(n+'_raw.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),v[0],(1600,1000)) for n,v in specs.items()}
    cap=cv2.VideoCapture(str(SOURCE/'source.mp4'))
    for k in range(300):
        ok,im=cap.read();assert ok
        canvas=np.full((1000,1600,3),BG,np.uint8)
        text(canvas,'MAP-ONLY FILTER / SAME DETECTIONS, SAME IDs',24,36,.77,INK,2)
        text(canvas,f'+{k/25:05.2f}s   |   Boxes, identities, visibility flags and calibration are frozen',24,65,.49,MUTED)
        base=data['current'];rows=base['frames'][k]
        annotated=source_frame(im,rows,base['diagnostics'][k]['unassigned_high_confidence_boxes'])
        source=cv2.resize(annotated[275:675,:],(752,235))
        counts=Counter(o['status'] for o in rows)
        for col,mode in enumerate(('current','kalman')):
            x=24+800*col
            text(canvas,'CURRENT / existing map smoothing' if col==0 else 'FILTERED / gated Kalman + median',x,101,.59,INK,2)
            text(canvas,f"Detected: {counts['observed']}   Predicted: {counts['predicted']}   Unknown: {counts['unknown']}",x,128,.45,MUTED)
            canvas[160:395,x:x+752]=source
            text(canvas,'CURRENT STAGE MAP' if col==0 else 'KALMAN STAGE MAP',x,447,.49,INK)
            cnt=draw_map(canvas,x+51,481,data[mode]['frames'],k,labels.get(str(k),[]))
            text(canvas,f"{cnt['proxy']} solid proxies   |   {cnt['estimated']} hollow estimates",x+51,932,.45,MUTED)
            if col:
                events=Counter(o.get('filter_event','') for o in data[mode]['frames'][k])
                text(canvas,f"Rejected jumps this frame: {events['jump_rejected']}",x+51,954,.40,MUTED)
        text(canvas,'Solid = filtered floor proxy. Hollow = prediction, max 1s. White crosses = reviewed references on labeled frames.',24,977,.42,MUTED)
        text(canvas,'Both source panels are identical. Both maps use the same maximum step. Filtering cannot repair identity swaps.',24,997,.41,MUTED)
        for name,(fps,start,end) in specs.items():
            if start<=k<=end:writers[name].write(canvas)
        if k in [0,50,100,138,195,225,255,285,299]:cv2.imwrite(str(OUT/f'comparison_{k:03}.jpg'),canvas)
    cap.release()
    for w in writers.values():w.release()
    import imageio_ffmpeg
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    for name in specs:
        subprocess.run([ff,'-y','-i',str(OUT/(name+'_raw.mp4')),'-c:v','libx264','-crf','20','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/(name+'.mp4'))],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    print('Rendered three synchronized comparison videos',flush=True)

def plots(data,labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,9),layout='constrained')
    for row,(ident,start,end) in enumerate([(8,0,149),(13,0,149),(20,225,299)]):
        for col in range(2):
            ax=axes[row,col]
            for mode,color in [('current','#337cb5'),('kalman','#cb552b')]:
                tt=[];pp=[]
                for k in range(start,end+1):
                    o=data[mode]['frames'][k][ident-1];tt.append(k/25);pp.append(o['map'][col] if o.get('map') is not None else np.nan)
                ax.plot(tt,pp,label=mode.capitalize(),color=color,lw=1.6)
            refs=[]
            for key,rr in labels.items():
                k=int(key);o=next(o for o in rr if o['id']==ident)
                if start<=k<=end and o['floor'] is not None:refs.append((k/25,stage(o['floor'])[col]))
            if refs:
                rr=np.array(refs);ax.scatter(rr[:,0],rr[:,1],marker='x',c='#222222',s=26,label='Reviewed support',zorder=5)
            ax.set_title(f'Dancer {ident} / '+('across stage' if col==0 else 'stage depth'))
            ax.set_xlabel('Clip time (seconds)');ax.set_ylabel('% of mat width');ax.grid(alpha=.2)
            if row==0 and col==0:ax.legend(fontsize=9)
    fig.suptitle('Timing and position: real clip, same identities\nReviewed points are sparse and approximate; gaps are not filled with reference labels.',fontsize=13)
    fig.savefig(OUT/'motion_traces.png',dpi=140);plt.close(fig)

def report():
    scores=read(OUT/'accuracy.json')['development_final'];motion=read(OUT/'motion.json')['metrics'];stationary=read(OUT/'stationary.json');delay=read(OUT/'delay.json')
    a,b=scores['current'],scores['kalman'];ma,mb=motion['current'],motion['kalman']
    rows=[('Correct floor positions / 120',str(a['floor_correct_within_3pct']),str(b['floor_correct_within_3pct'])),('Median floor error (% mat width)',f"{a['median_floor_error_pct']:.2f}",f"{b['median_floor_error_pct']:.2f}"),('Abrupt changes in map step (>0.5% width/frame)',str(ma['abrupt_step_changes_over_half_pct']),str(mb['abrupt_step_changes_over_half_pct'])),('Large map steps (>0.5% width/frame)',str(ma['steps_over_half_pct_mat_width']),str(mb['steps_over_half_pct_mat_width'])),('Stationary fluctuation (% mat width)',f"{stationary['current']['median_rms_pct_width']:.3f}",f"{stationary['kalman']['median_rms_pct_width']:.3f}"),('Correct original-ID matches / 145',str(a['correct_initial_ids']),str(b['correct_initial_ids']))]
    table=''.join(f'<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td></tr>' for r in rows)
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Map-only Kalman test</title><style>
body{{background:#101619;color:#edf1f4;font:17px/1.65 system-ui;margin:0}}main{{max-width:1120px;margin:auto;padding:44px 24px 72px}}h1{{font-size:46px;line-height:1.1;margin:18px 0}}h2{{font-size:25px;margin-top:34px}}.tag{{color:#96d4c7;letter-spacing:2px;font-size:13px;text-transform:uppercase}}.card{{background:#1c282d;padding:24px;border-radius:14px;margin:24px 0}}video,img{{width:100%;border-radius:10px}}button{{padding:12px 16px;background:#293e47;color:#edf1f4;border:1px solid #526d79;border-radius:8px;margin:0 8px 12px 0;cursor:pointer}}a{{color:#9fcefa}}.muted{{color:#acbbc3}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #3c4d55}}th{{color:#96d4c7}}p{{max-width:940px}}</style><main>
<div class="tag">Dance project · map-only experiment · 300 frames</div><h1>Kalman filtering did not<br>reduce jumps overall.</h1><p>The detections and identities are exactly the same on both sides. The Kalman version reaches a few more reviewed floor positions, but moves more abruptly and has slightly more stationary jitter. Keep the current map filter for now.</p>
<div class="card"><button onclick="play('map_comparison.mp4')">Full clip · normal speed</button><button onclick="play('movement_half_speed.mp4')">First 6 seconds · half speed</button><button onclick="play('overlap_half_speed.mp4')">Final 3 seconds · half speed</button><video id="video" src="map_comparison.mp4" poster="comparison_138.jpg" controls playsinline preload="metadata"></video><p class="muted">Left: current median + bounded smoothing. Right: median + gated Kalman filter. Source video, bounding boxes, IDs and calibration are identical. Solid dots are floor proxies; hollow dots are temporary predictions. White crosses are reviewed references on labeled frames.</p></div>
<h2>What changed</h2><p>Only the map calculation changes. The Kalman filter estimates position and velocity, rejects unusually large prediction errors, and admits a persistent change after three consistent observations. It has the same maximum displayed step as the current map: 0.875% of mat width per frame. Predictions last at most one second. No retraining, new labels, new boxes or ID reassignment.</p>
<h2>Measured tradeoff</h2><table><thead><tr><th>Check</th><th>Current</th><th>Kalman</th></tr></thead><tbody>{table}</tbody></table>
<p class="muted">Position and identity scores use eight previously reviewed final-section frames. Position tolerance is 3% of mat width. Step metrics use {ma['paired_transitions']} shared transitions across the full clip; a large step is not automatically an error. Abruptness is the change between consecutive displacement vectors, measured on shared three-frame sequences. The stationary check uses {len(stationary['paired_ids'])} common dancers in the final 0.56 seconds and excludes hollow predictions.</p>
<h2>Did it delay real movement?</h2><p>There is no clear evidence of extra average delay in this clip, but this test cannot establish zero lag. The exploratory comparison against interpolated reviewed points found {delay['usable_segments']} usable trajectory segments and a median change of {delay['median_added_lag_ms']:.0f} ms relative to the current map. Negative values mean earlier output. Some estimates imply leading the reference, which can also be prediction overshoot or imperfect labels. Treat this number as diagnostic, not a verified latency result.</p>
<p>This section mostly contains movement within a formation, with limited travel across the stage. The half-speed videos and traces below are more useful for inspecting catches, overshoot and delayed settling. Sparse annotations do not provide frame-accurate ground truth for a fast stage crossing.</p><img src="motion_traces.png" alt="Current and Kalman map trajectories compared with sparse reviewed floor positions">
<h2>Why the filter can make it worse</h2><p>The input is still a box-bottom proxy. A change in pose or a partial box can move that proxy while the dancer remains in place. A velocity model can interpret that as real travel and continue in the wrong direction. Rejecting a spike helps; accepting several consistent wrong positions does not fix their meaning. Identity swaps also remain identical because this experiment deliberately leaves tracking unchanged.</p>
<div class="card"><strong>Decision: retain the current filter.</strong><p>The map-only Kalman version does not meet the goal of reducing jumps without a timing tradeoff. It is saved as a comparison, not installed as the new default. The next useful improvement is a more reliable floor-location input or explicit rejection of uncertain support points, rather than stronger smoothing alone.</p></div>
<h2>How this was tested</h2><p>We first tested three Kalman response settings on raw frozen box-bottom coordinates. They increased fluctuation, so we also tested the same three settings with a short median prefilter. The selected setting preserves validation position accuracy within two matches and minimizes abrupt motion in the validation window. All six results are saved. This is iterative development on an already reviewed clip, not an independent fresh benchmark.</p>
<p><a href="Map_Filter_Pilot.ipynb">Concise notebook</a> · <a href="accuracy.json">Position scores</a> · <a href="motion.json">Map movement checks</a> · <a href="delay.json">Timing diagnostics</a> · <a href="selection.json">All settings</a> · <a href="provenance.json">Frozen-input verification</a></p>
</main><script>function play(src){{const v=document.getElementById('video');v.src=src;v.load();v.play();}}</script></html>'''
    (OUT/'START_HERE.html').write_text(body,encoding='utf-8')
    import nbformat as nbf
    nb=nbf.v4.new_notebook();nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}
    nb.cells=[nbf.v4.new_markdown_cell('# Map-only Kalman pilot\n\nQuestion: can we reduce map jumps without changing detection or delaying genuine motion?\n\nResult: this version does not improve overall stability. Keep the current map filter.'),nbf.v4.new_code_cell("import json\nfrom pathlib import Path\nfrom IPython.display import Video, Image, display\nroot = Path.cwd()\nload = lambda name: json.loads((root / name).read_text())\nload('provenance.json')"),nbf.v4.new_markdown_cell('## The comparison\n\nLeft: the existing map. Right: a causal position-and-velocity Kalman filter with innovation gating, a short median prefilter, and the same maximum map step. Boxes, dancer IDs, visibility eligibility and calibration are frozen. No labels enter filtering.'),nbf.v4.new_code_cell("display(Video(str(root / 'map_comparison.mp4'), embed=False, width=1000))"),nbf.v4.new_markdown_cell('## Results\n\nAbrupt-step counts describe movement of displayed dots, not proof that every such step is an error. Floor accuracy uses sparse approximate references. Both methods must keep the same detection and identity scores.'),nbf.v4.new_code_cell("a = load('accuracy.json')['development_final']\nm = load('motion.json')['metrics']\nfor mode in ['current', 'kalman']:\n    print(mode, 'floor matches:', a[mode]['floor_correct_within_3pct'], '/ 120', 'abrupt step changes:', m[mode]['abrupt_step_changes_over_half_pct'])\nload('stationary.json')"),nbf.v4.new_markdown_cell('## Movement timing\n\nThe reviewed points are too sparse to establish zero delay. The lag scan uses interpolation only for evaluation, removes constant position bias, and excludes flat or boundary optima. Negative lags can reflect overshoot or label error. Inspect the actual replay and trajectories.'),nbf.v4.new_code_cell("display(Image(filename=str(root / 'motion_traces.png')))\nd = load('delay.json')\n{k: v for k, v in d.items() if k != 'segments'}"),nbf.v4.new_markdown_cell('## Decision\n\nThe Kalman model improved the number of floor matches slightly, but increased abrupt changes and stationary fluctuation. It did not repair identity swaps. The existing filter remains the default. Better floor-contact input is the next priority.\n\nSix settings were explored during development; see `selection.json`. This is a previously reviewed clip, not a fresh holdout. No broad labeling effort is justified by this result.\n\nReproduce from the DL project root: run `Dance_Floor_Contact/map_kalman_pilot.py`, then `Dance_Floor_Contact/map_kalman_deliver.py` in the existing Python environment. The prior project artifacts remain required.')]
    from nbclient import NotebookClient
    NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(OUT)}}).execute();nbf.write(nb,OUT/'Map_Filter_Pilot.ipynb')
    for name in ['map_kalman_pilot.py','map_kalman_deliver.py']:shutil.copy2(Path(__file__).parent/name,OUT/name)

def verify(data):
    assert frozen_hash(data['current'])==frozen_hash(data['kalman'])
    for row in data['kalman']['frames']:
        for o in row:
            if o['map_status']=='estimated':assert 0<o['map_age_frames']<=25
            if o['map_status']=='unknown':assert o['map'] is None
    results={'same_boxes_ids_status':True}
    for name,count,fps in [('map_comparison',300,25),('movement_half_speed',150,12.5),('overlap_half_speed',75,12.5)]:
        cap=cv2.VideoCapture(str(OUT/(name+'.mp4')));actualfps=cap.get(cv2.CAP_PROP_FPS);n=0
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape==(1000,1600,3);n+=1
        cap.release();assert n==count and actualfps==fps;results[name]={'decoded_frames':n,'fps':actualfps}
    write('verification.json',results);print(json.dumps(results),flush=True)

if __name__=='__main__':
    data=read(OUT/'tracks.json');labels=read(LABEL_DIR/'labels.json')
    render(data,labels);plots(data,labels);report();verify(data)
