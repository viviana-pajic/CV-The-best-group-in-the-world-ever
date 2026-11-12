"""Report figures from saved results; never reruns training or selects models."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent;FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white'})
COLORS=['#738393','#d69a45','#258f79']

def main():
    result=read(OUT/'reserved_evaluation/results.json');selection=read(OUT/'search/frozen_selection.json');baseline=result['baseline']['all'];variants=[baseline,result['candidates']['causal']['end_to_end']['all'],result['candidates']['offline']['end_to_end']['all']];labels=['Previous box-bottom map','Causal support pipeline','Offline support pipeline'];xx=np.arange(4)
    fig,ax=plt.subplots(figsize=(10,4.7))
    for j,(row,name,color) in enumerate(zip(variants,labels,COLORS)):
        vals=[100*row['correct_by_tolerance_pct_width'][str(t)]/row['visible_references'] for t in [1,2,3,5]];bars=ax.bar(xx+(j-1)*.25,vals,.23,label=name,color=color)
        for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.0f}%',ha='center',fontsize=8)
    ax.set_xticks(xx,['1%','2%','3%','5%']);ax.set_ylim(0,108);ax.set_ylabel('Correct / all visible reference points (%)');ax.set_xlabel('Allowed stage error, as a percentage of mat width');ax.set_title('Reserved section: accuracy includes missing positions');ax.legend(loc='upper left',fontsize=9);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();fig.savefig(FIG/'reserved_accuracy.png',dpi=160);plt.close(fig)
    ranking=read(OUT/'candidates/clean_ranking.json');best={}
    for r in ranking:
        suite=r['run'].split('_')[0]
        if suite not in best or r['validation']['selected']['joint_f1']>best[suite]['validation']['selected']['joint_f1']:best[suite]=r
    suite_names={'v1':'Small balanced set','v2':'More postures','v3':'Target-box channel','v4':'Local contact evidence','v5':'Larger, cleaner set'};fig,ax=plt.subplots(figsize=(10,4.5));keys=sorted(best)
    values=[100*best[k]['validation']['selected']['joint_f1'] for k in keys];bars=ax.barh([suite_names[k] for k in keys],values,color=['#95a5b4']*4+['#258f79'])
    for b,k,v in zip(bars,keys,values):
        s=read(OUT/'runs'/best[k]['run']/'settings.json');ax.text(v+.5,b.get_y()+b.get_height()/2,f"{v:.1f} · {s['training_boxes']} training crops",va='center',fontsize=9)
    ax.set_xlim(0,117);ax.set_xlabel('Best development joint F1 (%)');ax.set_title('Model search on the same clean validation labels');ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();fig.savefig(FIG/'training_comparison.png',dpi=160);plt.close(fig)
    cfg=selection['offline']['chosen']['support'];members=cfg.get('members',[cfg]);fig,axes=plt.subplots(1,len(members),figsize=(6.5*len(members),4),squeeze=False)
    for ax,m in zip(axes[0],members):
        hist=read(OUT/'runs'/m['run']/'history.json');epochs=[r['epoch'] for r in hist];ax.plot(epochs,[r['train_loss'] for r in hist],label='Training loss',color='#738393');ax.plot(epochs,[r['validation']['loss'] for r in hist],label='Validation loss',color='#258f79');ax.axvline(m['epoch'],ls='--',color='#c48840',label='Chosen checkpoint');ax.set_title(m['run']);ax.set_xlabel('Epoch');ax.set_ylabel('Combined loss');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(FIG/'selected_training_curves.png',dpi=160);plt.close(fig)
    geom=read(OUT/'diagnostics/geometry_sensitivity.json')['rows'];fig,ax=plt.subplots(figsize=(8,4));ax.plot([r['image_y'] for r in geom],[r['ten_pixels_above_floor_error_pct'] for r in geom],marker='o',color='#b77837');ax.set_xlabel('Image Y coordinate (back of stage → front)');ax.set_ylabel('Stage error (% mat width)');ax.set_title('A point 10 pixels above the floor shifts the stage map');ax.grid(alpha=.15);fig.tight_layout();fig.savefig(FIG/'perspective_sensitivity.png',dpi=160);plt.close(fig)
    # Keep every unrounded aggregate available for review outside the charts.
    (FIG/'tables.json').write_text(json.dumps({'reserved_comparison':dict(zip(labels,variants)),'suite_best':best},indent=2))
    print('Report figures written',flush=True)

if __name__=='__main__':main()
