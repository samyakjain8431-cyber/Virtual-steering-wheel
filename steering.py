import math
from config import DEADZONE_DEG, MAX_STEER_DEG


def compute_steering(left_wrist, right_wrist, neutral_angle, smooth_angle,
                     analog_x, stick_alpha=0.3, angle_alpha=0.1):
    """
    Compute steering values from two wrist positions.

    Parameters
    ----------
    left_wrist   : (x, y) pixel coords of left wrist
    right_wrist  : (x, y) pixel coords of right wrist
    neutral_angle: calibrated neutral angle (float or None)
    smooth_angle : current smoothed angle
    analog_x     : current analog stick X value
    stick_alpha  : smoothing factor for analog stick
    angle_alpha  : smoothing factor for raw angle

    Returns
    -------
    dict with keys: smooth_angle, neutral_angle, analog_x,
                    steering_angle, direction, center
    """
    dx = right_wrist[0] - left_wrist[0]
    dy = right_wrist[1] - left_wrist[1]
    angle = math.degrees(math.atan2(dy, dx))

    smooth_angle = angle_alpha * angle + (1 - angle_alpha) * smooth_angle

    if neutral_angle is None:
        neutral_angle = smooth_angle

    steering_angle = smooth_angle - neutral_angle

    # Deadzone + analog mapping
    if abs(steering_angle) < DEADZONE_DEG:
        raw_x = 0.0
    else:
        sign = 1.0 if steering_angle > 0 else -1.0
        effective = abs(steering_angle) - DEADZONE_DEG
        usable_range = MAX_STEER_DEG - DEADZONE_DEG
        raw_x = sign * (effective / usable_range)

    raw_x = max(-1.0, min(1.0, raw_x))
    target_x = -raw_x  # invert: positive angle = left turn = negative stick X
    analog_x = stick_alpha * target_x + (1 - stick_alpha) * analog_x

    # Direction label
    if steering_angle > DEADZONE_DEG:
        direction = "LEFT"
    elif steering_angle < -DEADZONE_DEG:
        direction = "RIGHT"
    else:
        direction = "STRAIGHT"

    center = (
        (left_wrist[0] + right_wrist[0]) // 2,
        (left_wrist[1] + right_wrist[1]) // 2,
    )

    return {
        'smooth_angle': smooth_angle,
        'neutral_angle': neutral_angle,
        'analog_x': analog_x,
        'steering_angle': steering_angle,
        'direction': direction,
        'center': center,
    }
