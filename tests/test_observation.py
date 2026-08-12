"""Frame preprocessing and the rolling 4-frame stack.

These exercise muzero.initial_frame_stack / push_frame directly. An earlier
version of this file asserted the same properties against inline np.tile
expressions, which meant reverting the production code to np.repeat left every
test passing -- caught by mutation testing.
"""
import numpy as np
import pytest

from muzero import IMAGE_RESHAPE, initial_frame_stack, preprocess_frame, push_frame

RAW_FRAME_SHAPE = (210, 160, 3)  # what MsPacman-v4 returns
STACKED = 4


def fake_frame(fill):
    return np.full(RAW_FRAME_SHAPE, fill, dtype=np.uint8)


def rgb_frame(r, g, b):
    """A (3,H,W) frame whose planes are constant, so layout is readable back."""
    return np.stack([np.full(IMAGE_RESHAPE, v, dtype=np.uint8) for v in (r, g, b)])


def channel_signature(stack):
    """First pixel of each channel plane -- enough to read back the layout."""
    return [int(stack[i, 0, 0]) for i in range(stack.shape[0])]


# --------------------------------------------------------------------------
# preprocess_frame
# --------------------------------------------------------------------------
def test_preprocess_gives_channels_first_rgb():
    assert preprocess_frame(fake_frame(120), resize_shape=IMAGE_RESHAPE).shape == (3, *IMAGE_RESHAPE)


def test_preprocess_keeps_colour():
    """The grayscale conversion was removed when the networks moved to 12 input
    channels; if it comes back the stack silently becomes the wrong shape."""
    frame = np.zeros(RAW_FRAME_SHAPE, dtype=np.uint8)
    frame[..., 0] = 200  # red only
    out = preprocess_frame(frame, resize_shape=IMAGE_RESHAPE)
    assert out[0].max() > 150 and out[1].max() == 0 and out[2].max() == 0


# --------------------------------------------------------------------------
# initial_frame_stack
# --------------------------------------------------------------------------
def test_initial_stack_has_the_network_input_shape():
    stack = initial_frame_stack(preprocess_frame(fake_frame(50), IMAGE_RESHAPE))
    assert stack.shape == (12, *IMAGE_RESHAPE)  # matches input_channels=12


def test_initial_stack_repeats_whole_frames():
    """Regression: np.repeat(frame, 4, axis=0) interleaves the channel planes
    into [R,R,R,R,G,G,G,G,B,B,B,B], but push_frame treats the stack as four
    consecutive RGB frames. The first observation of every episode was
    malformed, and stayed malformed for four steps as it shifted through."""
    assert channel_signature(initial_frame_stack(rgb_frame(10, 20, 30))) == [10, 20, 30] * 4


def test_repeat_would_produce_a_different_layout():
    """Pins why tile is required, so the distinction is not lost to a refactor."""
    frame = rgb_frame(10, 20, 30)
    assert channel_signature(np.repeat(frame, 4, axis=0)) == [10] * 4 + [20] * 4 + [30] * 4
    assert channel_signature(initial_frame_stack(frame)) != channel_signature(np.repeat(frame, 4, axis=0))


@pytest.mark.parametrize("num_frames", [1, 2, 4, 8])
def test_initial_stack_scales_with_frame_count(num_frames):
    stack = initial_frame_stack(rgb_frame(1, 2, 3), num_frames=num_frames)
    assert stack.shape == (3 * num_frames, *IMAGE_RESHAPE)
    assert channel_signature(stack) == [1, 2, 3] * num_frames


# --------------------------------------------------------------------------
# push_frame
# --------------------------------------------------------------------------
def test_rolling_update_preserves_frame_grouping():
    stack = push_frame(initial_frame_stack(rgb_frame(10, 20, 30)), rgb_frame(11, 21, 31))
    assert stack.shape == (12, *IMAGE_RESHAPE)
    assert channel_signature(stack) == [10, 20, 30] * 3 + [11, 21, 31]


def test_oldest_frame_is_evicted_after_a_full_cycle():
    stack = initial_frame_stack(rgb_frame(0, 0, 0))
    for i in range(1, STACKED + 1):
        stack = push_frame(stack, rgb_frame(i, i * 10, i * 20))
    # After four pushes nothing of the original frame remains.
    assert channel_signature(stack) == [1, 10, 20, 2, 20, 40, 3, 30, 60, 4, 40, 80]


def test_stack_shape_is_invariant_across_many_steps():
    stack = initial_frame_stack(preprocess_frame(fake_frame(0), IMAGE_RESHAPE))
    for step in range(1, 20):
        stack = push_frame(stack, preprocess_frame(fake_frame(step * 5), IMAGE_RESHAPE))
        assert stack.shape == (12, *IMAGE_RESHAPE)
    assert stack[9:, 0, 0].tolist() == [95, 95, 95]  # newest frame is last


def test_push_frame_does_not_mutate_its_input():
    stack = initial_frame_stack(rgb_frame(10, 20, 30))
    before = stack.copy()
    push_frame(stack, rgb_frame(99, 99, 99))
    assert np.array_equal(stack, before)


def test_push_frame_handles_single_channel_frames():
    """Channel count is read from the incoming frame rather than hardcoded to 3,
    so a return to grayscale would not silently corrupt the stack."""
    gray = np.full((1, *IMAGE_RESHAPE), 7, dtype=np.uint8)
    stack = push_frame(initial_frame_stack(gray), np.full((1, *IMAGE_RESHAPE), 9, dtype=np.uint8))
    assert stack.shape == (4, *IMAGE_RESHAPE)
    assert channel_signature(stack) == [7, 7, 7, 9]
