import numpy as np
import cv2
from distance_frame_annotator.core import sample_rows,propagate,annotation_data,save_annotation

def test_sampling_gap_and_repeatability():
    rows=[dict(video='v',frame_index=i,distance_m=i/10) for i in range(200)]
    a=sample_rows(rows,10,4,20,42)
    assert a==sample_rows(rows,10,4,20,42)
    assert all(abs(x['frame_index']-y['frame_index'])>=20 for i,x in enumerate(a) for y in a[i+1:])

def test_propagation_translation():
    rng=np.random.default_rng(2);im=np.zeros((100,100,3),np.uint8);im[30:50,30:50]=rng.integers(0,256,(20,20,3),dtype=np.uint8)
    other=cv2.warpAffine(im,np.float32([[1,0,5],[0,1,3]]),(100,100));poly=[[[30,30],[49,30],[49,49],[30,49]]]
    draft,score=propagate(im,other,poly,20,.75)
    assert score>.99 and draft[0][0]==[35,33]

def test_annotation_area_and_coordinates(tmp_path):
    crop=np.zeros((30,40,3),np.uint8);row=dict(frame_index=1,roi='buoy',distance_m=12,bin_lower_m=10)
    record,mask=annotation_data(row,'v.mp4',[100,200,40,30],[[[2,2],[6,2],[6,6],[2,6]]],crop)
    assert record['union_area_pixels']==25
    assert record['objects'][0]['polygon_original'][0]==[102,202]
    key=save_annotation(tmp_path,record,crop,mask)
    assert (tmp_path/'masks'/(key+'.png')).exists()


def test_track_lab_import_matches_selected_video(tmp_path):
    from distance_frame_annotator.core import load_rows

    path = tmp_path / 'track.csv'
    path.write_text(
        'video_filename,frame_number,sample_or_frame,distance_m,mmsi\n'
        'a.mp4,0,100,12.5,123\n'
        'b.mp4,0,101,20,123\n'
        'a.mp4,,102,10,123\n'
        'a.mp4,3,103,nan,123\n',
        encoding='utf-8-sig',
    )
    rows = load_rows(path, 'ships', r'D:\videos\A.mp4')
    assert [(r['frame_index'], r['distance_m'], r['roi']) for r in rows] == [(0, 12.5, 'ships')]


def test_ocr_import_keeps_existing_filters(tmp_path):
    from distance_frame_annotator.core import load_rows

    path = tmp_path / 'ocr.csv'
    path.write_text(
        'video,frame_index,distance_m,roi,status\n'
        'v.mp4,1,12,ships,manual_accepted\n'
        'v.mp4,2,13,ships,rejected\n'
        'v.mp4,3,14,buoy,accepted\n',
        encoding='utf-8',
    )
    assert [r['frame_index'] for r in load_rows(path, 'ships', 'renamed.mp4')] == [1]


def test_distance_workspace_filters_before_sampling():
    from distance_frame_annotator.core import distance_workspace

    rows = [dict(video='v', frame_index=i, distance_m=i) for i in range(20)]
    assert distance_workspace(rows) == rows
    assert distance_workspace(rows, '', '1') == rows[:2]
    assert distance_workspace(rows, '18', '') == rows[18:]
    bounded = distance_workspace(rows, '8', '11')
    assert bounded == rows[8:12]
    assert {r['bin_lower_m'] for r in sample_rows(bounded, 10, 10, 0, 42)} == {0, 10}


def test_distance_workspace_rejects_invalid_bounds():
    import pytest
    from distance_frame_annotator.core import distance_workspace

    for lower, upper in [('nan', ''), ('', 'inf'), ('-1', ''), ('5', '4'), ('bad', '')]:
        with pytest.raises(ValueError):
            distance_workspace([], lower, upper)
