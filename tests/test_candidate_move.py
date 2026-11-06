from types import SimpleNamespace
import numpy as np
from pixel_annotator.history import MaskHistory
from distance_frame_annotator.integrated import App, Editor


class Var:
    def __init__(self, value):
        self.value = value
    def get(self):
        return self.value
    def set(self, value):
        self.value = value


def editor():
    obj = Editor.__new__(Editor)
    obj.owner = SimpleNamespace(mode=Var('current'))
    mask = np.zeros((12, 14), np.uint8)
    mask[3:6, 4:7] = 1
    mask[4, 5] = 0
    gt = np.zeros_like(mask)
    gt[10, 10] = 1
    obj.state = SimpleNamespace(ready=True, candidate_mask=mask, gt_mask=gt, image_rgb=np.zeros((12, 14, 3), np.uint8))
    obj._space_down = obj._pan_active = False
    obj._candidate_drag = None
    obj.tool = Var('candidate_move')
    obj.zoom = 2
    obj.view_x = obj.view_y = 0
    obj.canvas = SimpleNamespace(focus_set=lambda: None, configure=lambda **kwargs: None)
    obj.history = MaskHistory()
    obj.status_text = Var('')
    obj.refresh_canvas = lambda: None
    return obj


def event(x, y):
    return SimpleNamespace(x=x*2, y=y*2)


def test_drag_preserves_shape_gt_and_undo_redo():
    obj = editor()
    before, gt = obj.state.candidate_mask.copy(), obj.state.gt_mask.copy()
    obj._on_left_press(event(4, 3))
    obj._on_left_drag(event(6, 4))
    obj._on_left_drag(event(7, 5))
    obj._on_left_release(event(7, 5))
    moved = obj.state.candidate_mask.copy()
    expected = np.zeros_like(before)
    ys, xs = np.nonzero(before)
    expected[ys+2, xs+3] = 1
    assert np.array_equal(moved, expected)
    assert np.array_equal(obj.state.gt_mask, gt)
    obj.undo()
    assert np.array_equal(obj.state.candidate_mask, before)
    assert obj.history.undo(obj.state.candidate_mask, obj.state.gt_mask) is None
    obj.redo()
    assert np.array_equal(obj.state.candidate_mask, moved)


def test_bounds_no_pixel_loss_and_readonly():
    obj = editor()
    count = int(obj.state.candidate_mask.sum())
    obj._on_left_press(event(4, 3))
    obj._on_left_drag(event(-100, -100))
    assert obj.state.candidate_mask.sum() == count
    ys, xs = np.nonzero(obj.state.candidate_mask)
    assert xs.min() == 0 and ys.min() == 0
    obj._on_left_release(event(0, 0))
    obj.owner.mode.set('reference')
    obj._on_left_press(event(0, 0))
    assert obj._candidate_drag is None


def test_import_is_copy_and_keeps_gt():
    app = App.__new__(App)
    obj = editor()
    original = obj.state.candidate_mask.copy()
    obj.state.candidate_mask.fill(0)
    gt = obj.state.gt_mask.copy()
    app.active = {'roi': [0, 0, 14, 12]}
    app.last_saved = {'frame': 123, 'roi': app.active['roi'].copy(),
                      'mask': original.copy(), 'rgb': obj.state.image_rgb.copy()}
    app.editor, app.mode = obj, Var('current')
    app.ensure_applied = lambda: None
    app.update_status = lambda: None
    app.import_last_gt()
    assert np.array_equal(obj.state.candidate_mask, original)
    assert np.array_equal(obj.state.gt_mask, gt)
    assert obj.tool.get() == 'candidate_move'
    obj.state.candidate_mask.fill(0)
    assert np.array_equal(app.last_saved['mask'], original)
    assert np.array_equal(app.reference['mask'], original)
    assert app.proposal['reference_frame'] == 123
