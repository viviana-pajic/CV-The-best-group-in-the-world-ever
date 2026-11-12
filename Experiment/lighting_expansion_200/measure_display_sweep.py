"""Measure the display sweep and export the 0.30 replay's map coordinates."""
from collections import Counter

import numpy as np

from evaluate_trial import DEST, EXP, FIRST, export, inputs, read, trajectory_fit, write
from filter_lab import inside


def measure(tracks):
    frames = [
        {row['id']: row for row in frame if inside(row.get('map'))}
        for frame in tracks['frames']
    ]
    steps, accelerations = [], []
    toggles = flickers = 0
    for index, (previous, current) in enumerate(zip(frames, frames[1:]), start=1):
        toggles += len(previous.keys() ^ current.keys())
        common = previous.keys() & current.keys()
        for dancer in common:
            step = np.array(current[dancer]['map']) - previous[dancer]['map']
            steps.append(np.linalg.norm(step))
            if index + 1 < len(frames) and dancer in frames[index + 1]:
                next_step = np.array(frames[index + 1][dancer]['map']) - current[dancer]['map']
                accelerations.append(np.linalg.norm(next_step - step))
        if index + 1 < len(frames):
            following = frames[index + 1]
            flickers += len(current.keys() - (previous.keys() | following.keys()))
            flickers += len((previous.keys() & following.keys()) - current.keys())
    counts = Counter(row['map_status'] for frame in frames for row in frame.values())
    return {
        'frames': len(frames), 'displayed_dots': sum(counts.values()),
        'reported_dots': counts['proxy'], 'estimated_dots': counts['estimated'],
        'visibility_toggles': toggles, 'single_frame_flickers': flickers,
        'step_p99': float(np.percentile(steps, 99)),
        'step_max': float(max(steps)),
        'acceleration_p99': float(np.percentile(accelerations, 99)),
    }, frames


def main():
    run = DEST / 'runs/lighting_200_visual_s17'
    evaluation = run / 'evaluation'
    selection = read(evaluation / 'selection.json')
    base = read(EXP / 'late_full_replay/base_tracks.json')
    predictions = read(evaluation / 'predictions.json')
    sweep = []
    for threshold in (selection['threshold'], 0.75, 0.60, 0.50, 0.40, 0.30):
        points, confidence, ids = inputs(base, predictions, threshold)
        settings = selection['settings']
        fitted, states, ages = trajectory_fit.fit(points, confidence, settings)
        tracks = export(base, fitted, states, ages, points, ids, settings)
        stats, frames = measure(tracks)
        output = evaluation if threshold == selection['threshold'] else run / f'display_threshold_{threshold}'
        if output != evaluation:
            assert stats['displayed_dots'] == read(output / 'coverage.json')['display_dots']
        sweep.append({'threshold': threshold, **stats})
        if threshold == 0.30:
            write(output / 'map_positions.json', {
                'first_source_frame': FIRST,
                'fps': 25,
                'threshold': threshold,
                'bounds': {'x': [0, 100], 'y': [0, 62.5]},
                'axes': 'x increases left to right; y increases back to front',
                'fields': ['track_id', 'x', 'y', 'map_status', 'map_age_frames'],
                'status': {'proxy': 'reported contact', 'estimated': 'temporally estimated position'},
                'frames': [
                    [[dancer, *row['map'], row['map_status'], row['map_age_frames']]
                     for dancer, row in frame.items()]
                    for frame in frames
                ],
            })
    write(run / 'display_sweep.json', sweep)


if __name__ == '__main__':
    main()
