"""Port the earlier CVAT floor guide to the 200 fixed-camera images."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent))
from build_dataset import DEST, EXP, read, write

FLOOR = [
    {'name': 'mat_boundary', 'type': 'polyline', 'color': '#58f4d2', 'attributes': []},
    {'name': 'mat_rear_estimate', 'type': 'polyline', 'color': '#ffc96c', 'attributes': []},
    {'name': 'mat_grid', 'type': 'polyline', 'color': '#74bdfa', 'attributes': []},
]


def floor_lines(inverse):
    # Use the same five-by-five division and clipping as the earlier pilot.
    for i in range(6):
        s = i / 5
        side_label = 'mat_boundary' if i in (0, 5) else 'mat_grid'
        row_label = 'mat_rear_estimate' if i == 0 else 'mat_boundary' if i == 5 else 'mat_grid'
        for label, coords in [(side_label, [(s, 0), (s, 1)]),
                              (row_label, [(0, s), (1, s)])]:
            pixels = cv2.perspectiveTransform(np.array([coords], dtype=np.float64), inverse)[0]
            a, b = [tuple(map(int, np.rint(p))) for p in pixels]
            visible, a, b = cv2.clipLine((0, 0, 1280, 720), a, b)
            if visible:
                yield label, a, b


def main():
    calibration = read(EXP.parent/'Dance_CVAT_Pilot/calibration_draft.json')
    inverse = np.linalg.inv(calibration['pixel_to_relative_mat_homography'])
    segments = list(floor_lines(inverse))
    source = ET.parse(DEST/'cvat_export.xml')
    labels = source.find('meta/job/labels')
    if labels is None:
        labels = source.find('meta/task/labels')
    assert labels is not None and len(source.findall('image')) == 200
    labels.clear()
    for definition in FLOOR:
        info = ET.SubElement(labels, 'label')
        for field in ['name', 'type', 'color']:
            ET.SubElement(info, field).text = definition[field]
        ET.SubElement(info, 'attributes')
    for image in source.findall('image'):
        for child in list(image):
            image.remove(child)
        for name, a, b in segments:
            ET.SubElement(image, 'polyline', label=name, occluded='0', source='manual',
                          z_order='0', points=f'{a[0]},{a[1]};{b[0]},{b[1]}')
    ET.indent(source, space='  ')
    source.write(OUT/'floor_only_append.xml', encoding='utf-8', xml_declaration=True)
    write(OUT/'floor_schema.json', FLOOR)
    write(OUT/'floor_geometry.json', {'reference_source_frame': calibration['source_frame'],
          'geometry': segments, 'images': 200, 'floor_label_scope': 'Same fixed calibration and labels as the previous CVAT floor guide.'})
    print(len(segments), 'lines per frame on', len(source.findall('image')), 'frames')


if __name__ == '__main__':
    main()
