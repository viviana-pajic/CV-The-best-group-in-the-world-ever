"""Known-motion diagnostic: quantify smoothing tradeoffs without claiming video latency truth."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from filter_lab import *

def crossing(values,level,start=60,stop=180):
    for k in range(start+1,min(stop,len(values))):
        if np.isfinite(values[k-1:k+1]).all() and values[k-1]<=level<values[k]:
            return (k-1+(level-values[k-1])/max(values[k]-values[k-1],1e-9))/25
    return None

def main():
    selection=read(OUT/'search/frozen_selection.json');out=OUT/'diagnostics';out.mkdir(exist_ok=True);n=300;t=np.arange(n)/25;results={};draw=None
    for name,duration in [('gentle_motion',50),('brisk_motion',25),('fast_motion',10),('abrupt_measurement_step',1)]:
        truth=np.zeros((n,1,2));truth[:,:,0]=40;truth[:,:,1]=30;truth[:,0,0]+=10*np.clip((np.arange(n)-100)/duration,0,1)
        rng=np.random.default_rng(4729);raw=truth+rng.normal(0,.15,truth.shape);raw[165:167,0,1]+=25;raw[218:233]=np.nan;cf=np.where(np.isfinite(raw[:,:,0]),.9,0);rows={};curves={}
        for mode in ['causal','offline']:
            a,st,_=filtered(raw,cf,selection[mode]['chosen']['settings']);valid=np.isfinite(a[:,0,0]);err=np.linalg.norm(a[:,0]-truth[:,0],axis=1);cross=crossing(a[:,0,0],45);ref=crossing(truth[:,0,0],45);rows[mode]={'available_frames':int(valid.sum()),'estimated_frames':int((st[:,0]==2).sum()),'median_error_pct':float(np.nanmedian(err)),'p90_error_pct':float(np.nanpercentile(err,90)),'maximum_error_pct':float(np.nanmax(err)),'mid_movement_timing_difference_seconds':cross-ref if cross is not None and ref is not None else None,'outlier_window_max_error_pct':float(np.nanmax(err[160:190])),'pre_movement_displacement_pct':float(np.nanmax(np.abs(a[88:100,0,0]-40))),'near_stop_displacement_pct':float(np.nanmax(np.abs(a[100+duration:112+duration,0,0]-50)))};curves[mode]=a
        results[name]={'movement_duration_seconds':duration/25,'results':rows}
        if name=='brisk_motion':draw=(truth,raw,curves)
    write(out/'filter_response.json',{'scope':'Synthetic diagnostic only, not measured dancer latency. Same frozen filters; no parameter selection from these trajectories. Ten-percent mat-width movement, fixed0.15% Gaussian noise, two-frame25% corruption and15-frame missing interval. An instantaneous measurement step is intentionally nonphysical and tests rejection/recovery.','tests':results})
    truth,raw,curves=draw;plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    for j,ax in enumerate(axes):
        ax.plot(t,raw[:,0,j],'.',color='#aaa',markersize=2,label='Input with noise / two wrong frames');ax.plot(t,truth[:,0,j],color='#222',lw=2,label='Known motion')
        for mode,color in [('causal','#dd8844'),('offline','#258a76')]:ax.plot(t,curves[mode][:,0,j],color=color,lw=1.7,label=mode.title())
        ax.axvspan(218/25,233/25,color='#ddd',alpha=.4);ax.set_ylabel(('X' if j==0 else 'Y')+' (% mat width)');ax.grid(alpha=.15)
    axes[0].legend(fontsize=8,ncol=2);axes[-1].set_xlabel('Seconds');fig.suptitle('Filtering reduces errors, but can delay or spread real motion');fig.tight_layout();fig.savefig(out/'filter_response.png',dpi=160);plt.close(fig)
    print('Filter response diagnostic written')

if __name__=='__main__':main()
