"""Preview the second development-selected trajectory method and record its freeze."""
from render_comparison import *
import trajectory_fit,hashlib,datetime
from filter_characterize import crossing

def main():
    dest=OUT/'round2';selection=read(dest/'trajectory_search.json');cfg=selection['chosen']['settings'];tests={}
    for duration in [50,25,10,1]:
        truth=np.zeros((300,1,2));truth[:,:,0]=40;truth[:,:,1]=30;truth[:,0,0]+=10*np.clip((np.arange(300)-100)/duration,0,1);rng=np.random.default_rng(4729);raw=truth+rng.normal(0,.15,truth.shape);raw[165:167,0,1]+=25;raw[218:233]=np.nan;cf=np.where(np.isfinite(raw[:,:,0]),.9,0);a,st,_=trajectory_fit.fit(raw,cf,cfg);err=np.linalg.norm(a[:,0]-truth[:,0],axis=1);cc=crossing(a[:,0,0],45);tt=crossing(truth[:,0,0],45)
        tests[str(duration)]={'duration_seconds':duration/25,'p90_error_pct':float(np.nanpercentile(err,90)),'max_error_pct':float(np.nanmax(err)),'midpoint_timing_difference_seconds':cc-tt if cc is not None and tt is not None else None,'before_motion_displacement_pct':float(np.nanmax(np.abs(a[88:100,0,0]-40))),'near_stop_displacement_pct':float(np.nanmax(np.abs(a[100+duration:112+duration,0,0]-50)))}
    write(dest/'synthetic_response.json',tests);print('SYNTHETIC',tests,flush=True)
    baseline=read(OUT/'long_sequence/reference_tracks.json');candidate=read(dest/'previous_test_tracks.json')
    for d in [baseline,candidate]:d['frames']=d['frames'][380:];d['diagnostics']=d['diagnostics'][380:]
    labels={str(int(k)-4800):v for k,v in read(OUT/'data/holdout/labels.json').items()};pred=read(OUT/'reserved_evaluation/offline/predictions.json')[380:]
    render(baseline,candidate,OUT/'videos/reserved_source.avi',OUT/'videos/round2_diagnostic.mp4',labels,pred,selection['chosen']['threshold'],4800,'SMOOTHER TRAJECTORIES / DIAGNOSTIC REPLAY','Previously inspected section | Offline: uses future frames','Learned support + robust trajectory fit')
    write(dest/'frozen_selection.json',{'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection':selection['chosen'],'base_pipeline':selection['base_pipeline'],'script_sha256':hashlib.sha256((OUT/'code/trajectory_fit.py').read_bytes()).hexdigest(),'evaluation':'Backup5125..5265 remains unseen. Compare both round1 and round2 frozen settings there without choosing new parameters. Round1 reserved result remains historical; round2 on that same section is diagnostic only.'})

if __name__=='__main__':main()
