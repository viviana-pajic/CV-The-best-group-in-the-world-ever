"""Render a map display variant from cached floor contact predictions."""
import argparse

from evaluate_trial import DEST, EXP, export, inputs, read, render, trajectory_fit, write
from filter_lab import inside


def dot_count(tracks):
    return sum(inside(row.get('map')) for frame in tracks['frames'] for row in frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threshold', type=float, default=0.75)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error('--threshold must be between 0 and 1')

    run = DEST / 'runs' / 'lighting_200_visual_s17'
    evaluation = run / 'evaluation'
    selection = read(evaluation / 'selection.json')
    base = read(EXP / 'late_full_replay' / 'base_tracks.json')
    predictions = read(evaluation / 'predictions.json')
    points, confidence, ids = inputs(base, predictions, args.threshold)
    settings = selection['settings']
    fitted, states, ages = trajectory_fit.fit(points, confidence, settings)
    tracks = export(base, fitted, states, ages, points, ids, settings)
    output = run / f'display_threshold_{args.threshold}'
    output.mkdir(exist_ok=True)
    summary = {
        'frames': len(tracks['frames']),
        'reference_threshold': selection['threshold'],
        'display_threshold': args.threshold,
        'reference_dots': dot_count(read(evaluation / 'tracks.json')),
        'display_dots': dot_count(tracks),
    }
    print(summary, flush=True)
    render(output, tracks, title=f'Floor contact replay | threshold {args.threshold:g}',
           footer='Lower contact threshold admits less certain points. Same dancer detections and tracks.')
    write(output / 'coverage.json', summary)


if __name__ == '__main__':
    main()
