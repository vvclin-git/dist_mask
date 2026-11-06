import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image
from pixel_annotator.model import AnnotationState
from distance_frame_annotator.storage import AnnotationStore, binary, source_id, translate_proposal
from distance_frame_annotator.integrated import App, Editor
from pixel_annotator.history import MaskHistory
from pixel_annotator.app import AnnotationApp


def sample_state():
    state = AnnotationState()
    state.reset_for_image(Path('frame.png'), np.zeros((20, 30, 3), np.uint8))
    state.gt_mask[3:7, 4:9] = 1
    return state


def test_roundtrip_relocation_and_failed_commit(tmp_path, monkeypatch):
    video = tmp_path/'video.mp4'
    video.write_bytes(b'fake-video')
    moved = tmp_path/'moved.mp4'
    moved.write_bytes(video.read_bytes())
    identity = source_id(video)
    assert identity == source_id(moved)
    state = sample_state()
    row = dict(frame_index=0, roi='buoy', distance_m=34.6, bin_lower_m=30, bin_width_m=10)
    store = AnnotationStore(tmp_path/'output', identity, video)
    record = store.save(row, [5, 6, 30, 20], state.image_rgb, state, {})
    assert record['union_area_pixels'] == 20
    assert record['bbox_original'] == [9, 9, 5, 4]
    relocated = AnnotationStore(tmp_path/'output', identity, moved)
    mask, loaded = relocated.load(0, [5, 6, 30, 20])
    assert np.array_equal(mask, state.gt_mask)
    assert loaded['negative'] is False
    before = store.index.copy()
    def fail(*args, **kwargs):
        raise OSError('Simulated PNG write failure')
    monkeypatch.setattr(Image.Image, 'save', fail)
    with pytest.raises(OSError):
        store.save(row, [5, 6, 30, 20], state.image_rgb, state, {})
    assert store.index == before
    assert np.array_equal(store.load(0, [5, 6, 30, 20])[0], mask)


def test_legacy_and_corrupt_mask(tmp_path):
    folder = tmp_path/'annotations'
    folder.mkdir()
    data = dict(video='D:\\old\\v.mp4', frame_index=1, crop_roi=[0, 0, 30, 20],
                objects=[{'polygon_roi': [[4, 3], [8, 3], [8, 6], [4, 6]]}], union_area_pixels=20)
    (folder/'old.json').write_text(json.dumps(data))
    store = AnnotationStore(tmp_path, 'source', tmp_path/'v.mp4')
    assert np.count_nonzero(store.load(1, [0, 0, 30, 20])[0]) == 20
    masks = tmp_path/'masks'
    masks.mkdir()
    Image.fromarray(np.ones((20, 30), np.uint8)*255).save(masks/'old.png')
    with pytest.raises(ValueError, match='counts disagree'):
        store.load(1, [0, 0, 30, 20])
    assert binary(np.array([[0, 255]], np.uint8), (1, 2)).tolist() == [[0, 1]]


def test_mask_translation_and_failure_protection():
    rng = np.random.default_rng(9)
    rgb = np.zeros((70, 80, 3), np.uint8)
    rgb[20:40, 25:45] = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
    gt = np.zeros(rgb.shape[:2], np.uint8)
    gt[25:29, 30:34] = 1
    current = cv2.warpAffine(rgb, np.float32([[1, 0, 6], [0, 1, -3]]), (80, 70))
    original = gt.copy()
    proposal, meta = translate_proposal(rgb, current, gt, 15, .9)
    assert (meta['dx'], meta['dy']) == (6, -3)
    assert np.count_nonzero(proposal) == np.count_nonzero(gt)
    assert proposal[22:26, 36:40].sum() == 16
    assert np.array_equal(gt, original)
    with pytest.raises(ValueError):
        translate_proposal(rgb, np.zeros_like(rgb), gt, 15, .9)
    with pytest.raises(ValueError, match='empty'):
        translate_proposal(rgb, current, np.zeros_like(gt), 15, .9)


def test_sample_navigation_uses_cursor_not_nearest_frame():
    app = App.__new__(App)
    app.samples = [dict(frame_index=100), dict(frame_index=900), dict(frame_index=200)]
    app.cursor = 0
    app.current = 500
    app.ensure_applied = lambda: None
    app.sync_selection = lambda: None
    app.update_status = lambda: None
    targets = []
    app.show = lambda frame: targets.append(frame) or True
    app.next_sample(1)
    assert targets == [900] and app.cursor == 1
    app.show = lambda frame: False
    app.next_sample(1)
    assert app.cursor == 1


def test_empty_requires_explicit_negative(tmp_path):
    state = sample_state()
    state.gt_mask.fill(0)
    row = dict(frame_index=3, roi='buoy', distance_m=None, bin_lower_m=None, bin_width_m=10)
    store = AnnotationStore(tmp_path, 'source', 'v.mp4')
    with pytest.raises(ValueError):
        store.save(row, [0, 0, 30, 20], state.image_rgb, state, {})
    record = store.save(row, [0, 0, 30, 20], state.image_rgb, state, {}, negative=True)
    assert record['negative'] and record['bbox_roi'] is None


def test_install_frame_preserves_view_and_clears_working_state():
    app = App.__new__(App)
    class Var:
        def __init__(self, value=None):
            self.value = value
        def set(self, value):
            self.value = value
    editor = SimpleNamespace(state=sample_state(), _image_generation=0, history=MaskHistory(),
                             zoom=12.5, view_x=7.25, view_y=3.5,
                             _clear_palette_ui=lambda: None, refresh_canvas=lambda: None,
                             canvas=SimpleNamespace(configure=lambda **kwargs: None), status_text=Var())
    editor.history.push(editor.state.gt_mask, editor.state.gt_mask)
    app.editor = editor
    app.active = {'settings': {'video': 'v.mp4'}}
    app.v = {'frame': Var()}
    app.mode = Var()
    app.update_status = lambda: None
    newmask = np.zeros((20, 30), np.uint8)
    app.install(0, np.zeros((20, 30, 3), np.uint8), newmask, None)
    assert app.current == 0 and app.v['frame'].value == '0'
    assert (editor.zoom, editor.view_x, editor.view_y) == (12.5, 7.25, 3.5)
    assert not np.any(editor.state.candidate_mask)
    assert editor.history.undo(editor.state.candidate_mask, editor.state.gt_mask) is None
    assert editor._image_generation == 1


def test_reference_controls_do_not_edit_gt():
    dummy = SimpleNamespace(owner=SimpleNamespace(mode=SimpleNamespace(get=lambda: 'reference')),
                            status_text=SimpleNamespace(set=lambda text: None))
    # The editor guard returns before touching state or history.
    Editor.clear_gt(dummy)
    Editor.undo(dummy)
    Editor.close_polygon(dummy)


def test_fields_do_not_overwrite_each_other(tmp_path):
    state = sample_state()
    row = dict(frame_index=5, roi='buoy', distance_m=10, bin_lower_m=10, bin_width_m=10)
    buoy = AnnotationStore(tmp_path, 'same', 'v.mp4', 'buoy')
    buoy.save(row, [0, 0, 30, 20], state.image_rgb, state, {})
    ships = AnnotationStore(tmp_path, 'same', 'v.mp4', 'ships')
    mask, record = ships.load(5, [0, 0, 30, 20])
    assert record is None and np.count_nonzero(mask) == 0


def test_roi_change_invalidates_background_result():
    class State:
        ready = True
        image_rgb = np.zeros((10, 10, 3), np.uint8)
        palette = []
    app = SimpleNamespace(state=State(), _image_generation=4, _clear_palette_ui=lambda: None,
                          analyze_button=SimpleNamespace(configure=lambda **kwargs: None),
                          _analysis_running=True)
    AnnotationApp._reset_roi_analysis(app, True)
    assert app._image_generation == 5
    result = SimpleNamespace(cluster_map=np.ones((10, 10), np.int16), palette=['stale'])
    AnnotationApp._finish_color_analysis(app, result, 4, None)
    assert app.state.palette == [] and np.all(app.state.cluster_map == -1)
