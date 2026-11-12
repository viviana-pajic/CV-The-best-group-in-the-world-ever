"""Score one validation-selected checkpoint and render the static camera section."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np
import torch

from build_dataset import DEST, EXP, FIRST, LAST, read, video_frames, write

sys.path.insert(0, str(EXP/'code'))
from candidate_evaluate import load_model, oracle_scores, predict_aug
from evaluate_final_confirmation import score
from filter_lab import export, inputs
from occlusion_deliver import draw_map, source_frame
import frozen_holdout
import trajectory_fit


def replay(run, output, base, selection):
    cached = output/'predictions.json'
    if cached.exists():
        return read(cached)
    net, condition, _ = load_model(run/'best_joint.pt', 'cuda')
    result = []
    for frame, image in video_frames():
        observed = [r for r in base['frames'][frame-FIRST] if r['status'] == 'observed']
        result.append(predict_aug(net, image, observed, selection['mode'], condition))
        if (frame-FIRST) % 300 == 0:
            print('Support inference:', frame-FIRST, '/', LAST-FIRST+1, flush=True)
    assert len(result) == len(base['frames']) == LAST-FIRST+1
    write(cached, result)
    return read(cached)


def calibrate(run, mode, output):
    """Use the deployed crop averaging mode for development threshold selection."""
    dataset = read(EXP/'data/development_ending.json')
    net, condition, _ = load_model(run/'best_joint.pt', 'cuda')
    predictions = {}
    for clip, data in dataset.items():
        for k in data['val']:
            frame = data['source_first'] + k
            path = EXP.parent/'Dance_Project_Shared/dataset/frames'/f'frame_{frame:06}.png'
            image = cv2.imread(str(path))
            assert image is not None, path
            boxes = [r for r in data['labels'][str(k)] if r['box'] is not None]
            predictions[f'{clip}:{k}'] = {
                str(ident): row for ident, row in predict_aug(net, image, boxes, mode, condition).items()
            }
    result = oracle_scores(predictions, dataset)
    write(output/'development_calibration.json', result)
    return result['selected']['threshold']


def render(output, tracks, *, title='Lighting expansion trial | 200 training images',
           footer='Same frozen dancer detections and tracks. New support model. Offline map fit.'):
    temp = output/'lighting_trial_raw.mp4'
    target = output/'lighting_trial_replay.mp4'
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*'mp4v'), 25, (1600, 760))
    assert writer.isOpened()
    try:
        for frame, image in video_frames():
            i = frame-FIRST
            rows = tracks['frames'][i]
            annotated = source_frame(image, rows, [])
            canvas = np.full((760, 1600, 3), (28, 25, 23), np.uint8)
            cv2.putText(canvas, title, (24, 46),
                        0, .82, (240, 240, 240), 2, cv2.LINE_AA)
            cv2.putText(canvas, f'Frame {frame}  |  solid: reported contact  |  hollow: estimated position',
                        (24, 81), 0, .53, (212, 212, 212), 1, cv2.LINE_AA)
            canvas[120:615, 22:902] = cv2.resize(annotated, (880, 495))
            draw_map(canvas, 930, 150, tracks['frames'], i, [])
            cv2.putText(canvas, footer,
                        (24, 680), 0, .52, (212, 212, 212), 1, cv2.LINE_AA)
            writer.write(canvas)
            if i == 1200:
                cv2.imwrite(str(output/'replay_preview.jpg'), canvas)
    finally:
        writer.release()
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', str(temp),
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(target)], check=True)
    cap = cv2.VideoCapture(str(target))
    assert cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) == LAST-FIRST+1
    assert cap.get(cv2.CAP_PROP_FPS) == 25
    cap.release()
    temp.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='lighting_200_s17')
    parser.add_argument('--skip-video', action='store_true')
    parser.add_argument('--verified-revision', type=Path)
    args = parser.parse_args()
    if (DEST/'relabel_requested.txt').exists():
        if args.verified_revision is None:
            print('Pilot confirmation deferred while annotation revision is in progress.', flush=True)
            return
        assert args.name != 'lighting_200_s17', 'Rejected pilot stays deferred'
        verification = read(args.verified_revision/'cvat_roundtrip_check.json')
        assert verification['status'] == 'passed'
        manifest = args.verified_revision/'training_manifest.json'
        settings = read(DEST/'runs'/args.name/'settings.json')
        assert settings['data_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
        data = read(manifest)['lighting_200']
        assert data['provenance']['cvat_xml_sha256'] == verification['export_sha256']
        assert len(data['train']) == 200 and len(data['val']) == 11
        assert not set(data['train']).intersection(range(3938,4093))
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    run = DEST/'runs'/args.name
    assert (run/'complete.json').exists()
    output = run/'evaluation'
    output.mkdir(exist_ok=True)
    freeze = read(EXP/'final_confirmation/frozen_selection.json')
    mode = freeze['support']['mode']
    threshold = calibrate(run, mode, output)
    with (run/'best_joint.pt').open('rb') as checkpoint:
        checkpoint_sha = hashlib.file_digest(checkpoint, 'sha256').hexdigest()
    selection = {
        'checkpoint': 'best_joint.pt',
        'checkpoint_sha256': checkpoint_sha,
        'mode': mode, 'threshold': threshold,
        'settings': freeze['settings'],
        'scope': 'Checkpoint and threshold selected using only the 11 original development validation frames.',
    }
    selection_path = output/'selection.json'
    if selection_path.exists():
        assert read(selection_path) == selection, 'Use a new evaluation directory for a changed selection'
    else:
        write(selection_path, selection)
    base = read(EXP/'late_full_replay/base_tracks.json')
    pred = replay(run, output, base, selection)
    z, cf, ids = inputs(base, pred, threshold)
    fitted, states, age = trajectory_fit.fit(z, cf, selection['settings'])
    tracks = export(base, fitted, states, age, z, ids, selection['settings'])
    write(output/'tracks.json', tracks)
    frozen_holdout.D = EXP/'data/final_confirmation'
    label_path = frozen_holdout.D/'labels.json'
    sealed = read(frozen_holdout.D/'sealed_protocol.json')
    assert hashlib.sha256(label_path.read_bytes()).hexdigest() == sealed['label_sha256']
    labels = read(label_path)
    candidate = score(tracks, labels)
    baseline = read(EXP/'final_confirmation/results.json')['candidates']['Expanded support']
    write(output/'results.json', {'candidate': candidate, 'baseline': baseline, 'selection': selection,
          'comparison_scope': 'Same cached detections and tracks and offline trajectory filter. Seven reserved same-video images. Only the support model and its development-selected threshold differ.'})
    for name, result in [('Baseline', baseline), ('Lighting trial', candidate)]:
        row = result['all']
        print(name, row['correct_by_tolerance_pct_width'], 'of', row['visible_references'],
              'visible contacts;', row['hidden_measured'], 'of', row['hidden_references'],
              'hidden reported measured', flush=True)
    if not args.skip_video:
        render(output, tracks)


if __name__ == '__main__':
    main()
