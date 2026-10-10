import math
from config import FINGERTIP_IDS, FIST_THRESHOLD, OPEN_THRESHOLD, GESTURE_HOLD_FRAMES


def classify_hand_gesture(hand_landmarks, frame_w, frame_h):
    """
    Return 'FIST', 'OPEN', or 'NEUTRAL' for a single hand.

    Strategy
    --------
    Compute the ratio:
        avg_tip_dist / hand_scale
    where
        avg_tip_dist = mean Euclidean distance (px) from each fingertip to wrist
        hand_scale   = distance (px) from wrist (0) to middle-finger MCP (9)

    Small ratio → fingers curled → FIST
    Large ratio → fingers extended → OPEN
    """
    wrist = hand_landmarks[0]
    wx = wrist.x * frame_w
    wy = wrist.y * frame_h

    mid_mcp = hand_landmarks[9]
    scale = math.hypot(
        (mid_mcp.x * frame_w) - wx,
        (mid_mcp.y * frame_h) - wy
    )
    if scale < 1e-6:
        return 'NEUTRAL'

    tip_distances = []
    for tip_id in FINGERTIP_IDS:
        tip = hand_landmarks[tip_id]
        d = math.hypot(
            (tip.x * frame_w) - wx,
            (tip.y * frame_h) - wy
        )
        tip_distances.append(d)

    ratio = (sum(tip_distances) / len(tip_distances)) / scale

    if ratio < FIST_THRESHOLD:
        return 'FIST'
    elif ratio > OPEN_THRESHOLD:
        return 'OPEN'
    else:
        return 'NEUTRAL'


def update_gesture_state(frame_gesture, gesture_counter, current_gesture):
    """
    Apply hysteresis: update rolling counters and return the new committed gesture.

    Parameters
    ----------
    frame_gesture   : 'FIST' | 'OPEN' | 'NEUTRAL' — majority vote this frame
    gesture_counter : dict with keys 'FIST', 'OPEN', 'NEUTRAL'
    current_gesture : last committed gesture

    Returns
    -------
    (gesture_counter, current_gesture) — updated values
    """
    for g in ('FIST', 'OPEN', 'NEUTRAL'):
        if g == frame_gesture:
            gesture_counter[g] = min(gesture_counter[g] + 1, GESTURE_HOLD_FRAMES)
        else:
            gesture_counter[g] = max(gesture_counter[g] - 1, 0)

    if gesture_counter['FIST'] >= GESTURE_HOLD_FRAMES:
        current_gesture = 'FIST'
    elif gesture_counter['OPEN'] >= GESTURE_HOLD_FRAMES:
        current_gesture = 'OPEN'
    else:
        current_gesture = 'NEUTRAL'

    return gesture_counter, current_gesture


def majority_vote(gesture_votes):
    """Return the dominant gesture from a list of per-hand votes."""
    if not gesture_votes:
        return 'NEUTRAL'
    fist_count = gesture_votes.count('FIST')
    open_count = gesture_votes.count('OPEN')
    if fist_count > open_count:
        return 'FIST'
    elif open_count > fist_count:
        return 'OPEN'
    else:
        return 'NEUTRAL'
