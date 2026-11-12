"""Create and execute a short, portable experiment notebook from completed runs."""
import json,platform,sys
from zipfile import ZipFile,ZIP_DEFLATED
import nbformat
import xml.etree.ElementTree as ET
from nbclient import NotebookClient
import torch,ultralytics
from corrected_setup import OUT,read

def main():
    cells=[]
    md=lambda s:cells.append(nbformat.v4.new_markdown_cell(s))
    code=lambda s:cells.append(nbformat.v4.new_code_cell(s))
    md('''# Dance tracking: corrected-label pilot

We tested whether correcting a small set of labels could improve dancer detection and stage positions. The source is a 12-second clip with 20 known dancers. Some are completely hidden for part of the clip.

The comparison starts from pretrained COCO weights. It uses the same crops, mat calibration and tracker before and after training.''')
    code('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image, Video
ROOT = Path.cwd()
def read(name):
    return json.loads((ROOT / name).read_text())
summary = read('dataset_summary.json')
display(pd.DataFrame({k: summary[k] for k in ('train', 'val', 'test')}).T)''')
    md('''## 1. Correct the labels

We reviewed 36 keyframes. Each visible dancer has a box. A support point marks the grounded foot, the centre between grounded feet, or the knee support when kneeling. Unclear support has no coordinate. A completely hidden dancer has no detection box, but its identity stays in the annotation record.

The labels are approximate and assistant-reviewed. The starting frames reuse earlier contact marks, checked against the selected images. Later points were placed visually. The other 264 frames are replay frames, not training labels.''')
    code("display(Image(filename=str(ROOT / 'audit/000.jpg'), width=1000))\ndisplay(Image(filename=str(ROOT / 'audit/299.jpg'), width=1000))")
    md('''## 2. Keep a separate test period

Training uses 24 frames from the first 5.52 seconds. Validation uses four frames from 6.6–7.8 seconds. Testing uses eight frames from the final three seconds. Temporal gaps separate the periods.

This tests improvement on a later section of the same video. It does not establish performance on another choreography or camera.''')
    code("display(pd.DataFrame(read('manifest.json')).groupby('split').agg(frames=('frame','count'), first_frame=('frame','min'), last_frame=('frame','max'))) ")
    md('''## 3. Fine-tune the models

We trained a YOLO26m detector and a YOLO26m pose model with one custom support point. Both see three overlapping stage crops, resized to an input size of 960. The backbone's first ten layers are frozen.

The point model learns a coordinate and an observability score. Unknown points are encoded with visibility zero. Its custom OKS tolerance is 0.1; the previous one-point setup inherited a much broader default of 1.0. This pilot changes labels and training together, so it does not isolate the effect of that tolerance.

Validation selects checkpoints, confidence thresholds and the final pipeline. Test scores are computed after those choices are locked.''')
    code('''runs = {name: read(f'training_{name}.json') for name in ('detect','pose')}
display(pd.DataFrame({name: {'training_minutes': v['elapsed_seconds']/60,
                            'epochs_requested': v['epochs_requested'],
                            'validation_box_mAP50': v['validation']['metrics/mAP50(B)']}
                      for name,v in runs.items()}).T.round(3))
choices = read('operating_points.json')
print('Selected on validation:', choices['selected_pipeline'])
display(pd.DataFrame(choices['settings']).T)''')
    md('''## 4. Measure the whole result

Boxes are matched one-to-one at IoU 0.5. A floor point counts as correct if it is within 3% of the displayed mat width. Missing points remain in the denominator. Incorrect points, including outputs on hidden supports, reduce point precision.

The median error describes only matched outputs. It should always be read alongside coverage. The calibration is a fixed visual draft, so these are relative distances, not metres.''')
    code('''results = read('test_metrics.json')
rows = []
for name in ('baseline','tuned_box','pose','hybrid'):
    s = results[name]
    rows.append({'pipeline': name, 'box_precision': s['boxes']['precision'],
                 'box_recall': s['boxes']['recall'],
                 'correct_floor_points': s['floor']['correct_within_3pct_mat_width'],
                 'visible_floor_references': s['floor']['visible_references'],
                 'floor_coverage': s['floor']['correct_fraction_of_all_visible_references'],
                 'floor_precision': s['floor']['precision_within_3pct'],
                 'median_error_%_mat_width': s['floor']['median_error_pct_mat_width']})
display(pd.DataFrame(rows).set_index('pipeline').round(3))
display(pd.DataFrame(read('tracking_metrics.json')).T)''')
    md('''The table above uses validation-selected confidence levels. The following check holds detector confidence at 0.25. This separates the training tradeoff from confidence tuning. Full-clip tracking counts include training frames; the sampled identity checks use only test frames.''')
    code('''fixed = read('fixed_threshold_metrics.json')
display(pd.DataFrame({name: {'precision': fixed[name]['boxes']['precision'],
                           'recall': fixed[name]['boxes']['recall'],
                           'correct_floor_points': fixed[name]['floor']['correct_within_3pct_mat_width']}
                      for name in ('baseline','tuned_box')}).T.round(3))''')
    md('''## 5. Watch the held-out footage

The replay below compares the pretrained and fine-tuned detectors on the final three seconds, slowed to half speed. The learned-point comparison and full 12-second replays are in `START_HERE.html`. White crosses show reviewed references on labeled frames. Hollow dots represent short interpolated gaps. A question mark means no support estimate was accepted.''')
    code("display(Video(filename=str(ROOT / 'detector_test_comparison.mp4'), embed=True, width=1100))")
    md('''## 6. Decide what to label next

Use the detector recall and floor coverage together to judge the result. A smoother map with fewer correct dancers is not sufficient. Remaining ID changes and fully hidden people are separate tracking limitations.

The annotation format tested here is **person box + visible support point + unknown/occluded status**. Instance masks were not used for this training run. The corrected image annotations and CVAT schema are included for inspection before assigning more labeling work.''')
    code("print(read('result_summary.json')['verdict'])")
    md('''Implementation references: [Ultralytics custom pose labels and keypoint tolerances](https://docs.ultralytics.com/datasets/pose/) and [training configuration](https://docs.ultralytics.com/modes/train/). The local scripts and JSON files record the actual settings and measured results.''')
    notebook=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    path=OUT/'Corrected_Label_Pilot.ipynb';nbformat.write(notebook,path)
    NotebookClient(notebook,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(OUT)}}).execute()
    nbformat.write(notebook,path)
    env={'python':sys.version,'torch':torch.__version__,'ultralytics':ultralytics.__version__,'device':torch.cuda.get_device_name(0)}
    (OUT/'environment.json').write_text(json.dumps(env,indent=2))
    schema=[{'name':'dancer','type':'rectangle','color':'#32b7fa','attributes':[{'name':'dancer_id','input_type':'text','mutable':False,'default_value':'','values':[]},{'name':'contact','input_type':'select','mutable':True,'default_value':'hidden_or_uncertain','values':['visible','hidden_or_uncertain']}]},{'name':'floor_support','type':'points','color':'#ffcc33','attributes':[{'name':'dancer_id','input_type':'text','mutable':False,'default_value':'','values':[]}]}]
    (OUT/'cvat_schema.json').write_text(json.dumps(schema,indent=2))
    tree=ET.parse(OUT/'corrected_keyframes_cvat.xml')
    for box in tree.findall('.//box'):box.set('occluded','0')
    # Floor uncertainty is the contact attribute, not CVAT's person-occlusion flag.
    tree.write(OUT/'corrected_keyframes_cvat.xml',encoding='utf-8',xml_declaration=True)
    (OUT/'CVAT_IMPORT.txt').write_text('Create a separate image task using cvat_schema.json. Upload cvat_images.zip as task data. Import corrected_keyframes_cvat.xml using CVAT for images 1.1. The images are sparse keyframes, not consecutive video frames. manifest.json maps them to original video frames. labels.json preserves fully occluded identities that have no image box. These are assistant-reviewed pilot annotations, ready for inspection rather than a claim of perfect ground truth.\n')
    with ZipFile(OUT/'cvat_images.zip','w',ZIP_DEFLATED) as z:
        for f in (OUT/'images').glob('*.png'):z.write(f,f.name)
    with ZipFile(OUT/'corrected_annotations.zip','w',ZIP_DEFLATED) as z:
        for name in ('cvat_schema.json','CVAT_IMPORT.txt','cvat_images.zip','corrected_keyframes_cvat.xml','labels.json','manifest.json','dataset_summary.json','protocol.json','visual_edits.json'):z.write(OUT/name,name)
    print('Executed notebook and CVAT inspection bundle ready',flush=True)
if __name__=='__main__':main()
