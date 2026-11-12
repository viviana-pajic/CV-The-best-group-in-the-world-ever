"""Small teammate-only review archive, separate from the full experiment."""
from package_team import ROOT,OUT,copy_file,archive
import json
DEST=ROOT/'Dance_Labeling_Review_2026-11-10'
if __name__=='__main__':
    for p in (OUT/'CVAT_Review').rglob('*'):
        if p.is_file():copy_file(p,DEST/'CVAT_Review'/p.relative_to(OUT/'CVAT_Review'))
    copy_file(OUT/'figures/label_examples.jpg',DEST/'figures/label_examples.jpg')
    copy_file(OUT/'code/import_cvat_review.py',DEST/'code/import_cvat_review.py')
    copy_file(OUT/'code/merge_review_exports.py',DEST/'code/merge_review_exports.py')
    copy_file(OUT/'code/prepare_reviewed_training.py',DEST/'code/prepare_reviewed_training.py')
    (DEST/'START_HERE.html').write_text('''<!doctype html><meta charset="utf-8"><title>Dance labeling review</title><style>body{font:20px/1.65 system-ui;max-width:850px;margin:70px auto;padding:20px;background:#f4f3ee;color:#213c34}a{color:#16745e}</style><h1>Dance labeling review</h1><p><a href="CVAT_Review/GUIDE.html">Open the labeling guide and assignments</a></p><p>Start with the main development task: 56 training frames split 19 / 19 / 18, plus nine separate validation frames. Review the supplied annotations before adding more images.</p><p>The reserved evaluation folders must stay outside training. The six-image supplemental ending task belongs to a later diagnostic experiment and is optional; it is not included in those assignments.</p><p>Download and unzip this folder before opening the guide. Import each image ZIP into a separate CVAT task with its matching XML. These are approximate assistant-prepared labels for human review.</p>''',encoding='utf-8')
    (DEST/'CHECK_EXPORT.txt').write_text('From this folder, run:\npython code/import_cvat_review.py your_export.xml reviewed_labels.json\n\nThis converter uses the Python standard library only. It creates a new file and preserves the original export.\nRecord the reviewer name/date separately. Keep sample_split unchanged.\nThe rectangles are support-crop proposals, not a complete detector annotation set: add all missed distinguishable dancers before using them to retrain detection.\n',encoding='utf-8')
    start=DEST/'START_HERE.html';text=start.read_text(encoding='utf-8').replace('56 training frames split 19 / 19 / 18, plus nine separate validation frames','60 training frames split 20 / 20 / 20, plus eleven separate validation frames').replace('The six-image supplemental ending task belongs to a later diagnostic experiment and is optional; it is not included in those assignments.','The six-image supplemental_ending folder is an earlier separate export of images now included in the main task. Do not import it again when using the main task.');start.write_text(text,encoding='utf-8')
    text=text.replace('Start with the main development task: 60 training frames split 20 / 20 / 20, plus eleven separate validation frames.','Choose your separate A, B or C task pack in the guide. Each contains 20 training frames plus 4 / 4 / 3 validation frames. The packs do not overlap. The development folder is the combined master for an archive or single reviewer.')
    start.write_text(text,encoding='utf-8')
    archive(DEST,ROOT/(DEST.name+'.zip'))
