"""Honest later-data diagnostic previews, keeping the earlier frozen videos."""
from render_comparison import *
import time,copy

def main():
    dest=OUT/'ending_experiment/pipeline'
    while not (dest/'results.json').exists() or len(read(dest/'results.json').get('candidates',{}))<2:time.sleep(5)
    old=read(dest/'previous_support_full_tracks.json');new=read(dest/'expanded_support_full_tracks.json');op=read(OUT/'full_replay/support/predictions.json');npred=read(dest/'support/predictions.json');threshold=read(OUT/'ending_experiment/results.json')['chosen']['threshold'];oldthreshold=read(OUT/'round2/frozen_selection.json')['selection']['threshold']
    for first,last,name,label_folder in [(4800,5080,'expanded_support_diagnostic','holdout'),(5315,5560,'expanded_support_ending','ending')]:
        source=OUT/'videos'/f'{name}_source.avi';cap=cv2.VideoCapture(str(OUT/'full_replay/source.avi'));cap.set(cv2.CAP_PROP_POS_FRAMES,first-3250);w=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'FFV1'),25,(1280,720));assert w.isOpened()
        for k in range(last-first+1):ok,im=cap.read();assert ok;w.write(im)
        cap.release();w.release();left=copy.deepcopy(old);right=copy.deepcopy(new)
        for d in [left,right]:
            for key in ['frames','diagnostics']:d[key]=d[key][first-3250:last-3250+1]
        labels={str(int(k)-first):v for k,v in read(OUT/'data'/label_folder/'labels.json').items() if first<=int(k)<=last};render(left,right,source,OUT/'videos'/f'{name}.mp4',labels,npred[first-3250:last-3250+1],threshold,first,'EXTRA SEATED LABELS / SUPPORT MODEL DIAGNOSTIC','Same boxes and IDs | Ending replay includes training frames' if name.endswith('ending') else 'Previously inspected section | Development-selected new head','Expanded support model',ring_description='Rings: each model\'s raw support point. Dots: offline filtered positions. White crosses: approximate reference.',before_title='Previous support model',before_pred=op[first-3250:last-3250+1],before_threshold=oldthreshold)

if __name__=='__main__':main()
