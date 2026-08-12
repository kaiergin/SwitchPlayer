"""Frame preprocessing and the rolling 4-frame stack."""
import numpy as np
import pytest

from muzero import IMAGE_RESHAPE, preprocess_frame

RAW_FRAME_SHAPE = (210, 160, 3)  # what MsPacman-v4 returns


def fake_frame(fill):
    return np.full(RAW_FRAME_SHAPE, fill, dtype=np.uint8)


def channel_signature(stack):
    """First pixel of each channel plane -- enough to read back the layout."""
    return [int(stack[i, 0, 0]) for i in range(stack.shape[0])]


def test_preprocess_gives_channels_first_rgb():
    out = preprocess_frame(fake_frame(120), resize_shape=IMAGE_RESHAPE)
    assert out.shape == (3, *IMAGE_RESHAPE)


def test_preprocess_keeps_colour():
    """The grayscale conversion was removed when the networks moved to 12 input
    channels; if it comes back, the stack silently becomes the wrong shape."""
    frame = np.zeros(RAW_FRAME_SHAPE, dtype=np.uint8)
    frame[..., 0] = 200  # red only
    out = preprocess_frame(frame, resize_shape=IMAGE_RESHAPE)
    assert out[0].max() > 150 and out[1].max() == 0 and out[2].max() == 0


def test_initial_stack_repeats_whole_frames():
    """Regression: np.repeat(frame, 4, axis=0) interleaves channel planes into
    [R,R,R,R,G,G,G,G,B,B,B,B], but the rolling update below treats the stack as
    four consecutive RGB frames. np.tile is the correct construction."""
    frame = np.stack([np.full(IMAGE_RESHAPE, v, dtype=np.uint8) for v in (10, 20, 30)])
    stack = np.tile(frame, (4, 1, 1))
    assert stack.shape == (12, *IMAGE_RESHAPE)
    assert channel_signature(stack) == [10, 20, 30] * 4


def test_the_old_repeat_construction_was_wrong():
    frame = np.stack([np.full(IMAGE_RESHAPE, v, dtype=np.uint8) for v in (10, 20, 30)])
    assert channel_signature(np.repeat(frame, 4, axis=0)) == [10] * 4 + [20] * 4 + [30] * 4


def test_rolling_update_preserves_frame_grouping():
    old = np.stack([np.full(IMAGE_RESHAPE, v, dtype=np.uint8) for v in (10, 20, 30)])
    new = np.stack([np.full(IMAGE_RESHAPE, v, dtype=np.uint8) for v in (11, 21, 31)])
    stack = np.tile(old, (4, 1, 1))
    stack = np.concatenate([stack[3:, :, :], new], axis=0)
    assert stack.shape == (12, *IMAGE_RESHAPE)
    assert channel_signature(stack) == [10, 20, 30] * 3 + [11, 21, 31]


def test_stack_stays_12_channels_across_many_steps():
    stack = np.tile(preprocess_frame(fake_frame(0), IMAGE_RESHAPE), (4, 1, 1))
    for step in range(1, 20):
        nxt = preprocess_frame(fake_frame(step * 5), IMAGE_RESHAPE)
        stack = np.concatenate([stack[3:, :, :], nxt], axis=0)
        assert stack.shape == (12, *IMAGE_RESHAPE)
    # Newest frame occupies the last three planes.
    assert stack[9:, 0, 0].tolist() == [95, 95, 95]
