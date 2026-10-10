import cv2
import math
from config import HAND_CONNECTIONS


# ── Steering Wheel Overlay ─────────────────────────────────────────────────────

def draw_steering_wheel(frame, center, radius, rotation_deg, analog_x, direction):
    """
    Draw an animated virtual steering wheel on *frame* in-place.

    Parameters
    ----------
    center       : (cx, cy)  – pixel position of the wheel hub
    radius       : int       – outer radius in pixels
    rotation_deg : float     – current steering rotation (degrees)
    analog_x     : float     – stick value in [-1, +1], used for colour tint
    direction    : str       – 'LEFT' | 'RIGHT' | 'STRAIGHT'
    """
    cx, cy = center
    rot_rad = math.radians(rotation_deg)

    # Colour palette based on direction
    if direction == "LEFT":
        rim_color   = (255, 140,  40)
        spoke_color = (255, 180,  80)
        hub_color   = (200, 100,  20)
    elif direction == "RIGHT":
        rim_color   = ( 40, 140, 255)
        spoke_color = ( 80, 180, 255)
        hub_color   = ( 20, 100, 200)
    else:
        rim_color   = ( 50, 220,  80)
        spoke_color = ( 80, 255, 120)
        hub_color   = ( 30, 160,  50)

    # Semi-transparent backing disc
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), radius + 4, (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Outer rim
    rim_thickness = max(6, radius // 8)
    cv2.circle(frame, (cx, cy), radius, rim_color, rim_thickness)

    # 3-D bevel
    highlight = tuple(min(255, c + 80) for c in rim_color)
    shadow    = tuple(max(0,   c - 80) for c in rim_color)
    cv2.circle(frame, (cx, cy), radius,                    highlight, 1)
    cv2.circle(frame, (cx, cy), radius - rim_thickness + 1, shadow,   1)

    # Three spokes at 0°, 120°, 240°
    inner_r = radius // 5
    outer_r = radius - rim_thickness // 2
    spoke_w = max(3, radius // 12)
    for spoke_offset_deg in (0, 120, 240):
        angle = rot_rad + math.radians(spoke_offset_deg)
        x1 = int(cx + inner_r * math.cos(angle))
        y1 = int(cy + inner_r * math.sin(angle))
        x2 = int(cx + outer_r * math.cos(angle))
        y2 = int(cy + outer_r * math.sin(angle))
        cv2.line(frame, (x1, y1), (x2, y2), spoke_color, spoke_w, cv2.LINE_AA)

    # Hub cap
    hub_r = max(6, radius // 6)
    cv2.circle(frame, (cx, cy), hub_r,     hub_color,    -1)
    cv2.circle(frame, (cx, cy), hub_r,     highlight,     1)
    cv2.circle(frame, (cx, cy), hub_r - 3, (30, 30, 30), -1)

    # Steering-angle arc indicator
    arc_span = int(abs(analog_x) * 135)
    if arc_span > 2:
        start_angle = int(math.degrees(rot_rad)) - arc_span // 2
        end_angle   = start_angle + arc_span
        cv2.ellipse(
            frame, (cx, cy),
            (radius + rim_thickness // 2 + 3,
             radius + rim_thickness // 2 + 3),
            0, start_angle, end_angle,
            highlight, 2, cv2.LINE_AA
        )

    # Direction label
    label_y = cy + radius + rim_thickness + 18
    font_scale = max(0.45, radius / 90.0)
    cv2.putText(
        frame, direction,
        (cx - 30, label_y),
        cv2.FONT_HERSHEY_DUPLEX, font_scale,
        rim_color, 1, cv2.LINE_AA
    )


# ── Hand Skeleton & Landmarks ──────────────────────────────────────────────────

def draw_hand(frame, hand, w, h):
    """Draw skeleton connections and landmark dots for one hand."""
    for start, end in HAND_CONNECTIONS:
        x1 = int(hand[start].x * w)
        y1 = int(hand[start].y * h)
        x2 = int(hand[end].x * w)
        y2 = int(hand[end].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    for landmark in hand:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)


def draw_wrist(frame, wrist_x, wrist_y):
    """Highlight the wrist landmark."""
    cv2.circle(frame, (wrist_x, wrist_y), 7, (0, 255, 0), -1)


# ── HUD: Steering Info ─────────────────────────────────────────────────────────

def draw_steering_hud(frame, smooth_angle, steering_angle, direction, analog_x, w, h):
    """Draw angle, direction, and analog stick bar on frame."""
    cv2.putText(frame, f"Angle: {smooth_angle:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(frame, f"Steering Angle: {steering_angle:.2f}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(frame, f"Direction: {direction}",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Analog stick bar
    bar_cx = w // 2
    bar_y  = h - 30
    bar_half = 100
    fill_x = int(analog_x * bar_half)

    cv2.rectangle(frame, (bar_cx - bar_half, bar_y - 10),
                  (bar_cx + bar_half, bar_y + 10), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_cx, bar_y - 8),
                  (bar_cx + fill_x, bar_y + 8), (0, 200, 255), -1)
    cv2.rectangle(frame, (bar_cx - bar_half, bar_y - 10),
                  (bar_cx + bar_half, bar_y + 10), (200, 200, 200), 1)
    cv2.putText(frame, f"Stick X: {analog_x:+.2f}",
                (bar_cx - bar_half, bar_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


# ── HUD: Hand Info ─────────────────────────────────────────────────────────────

def draw_hand_info(frame, hand_name, thumb_tip, thumb_ip):
    """Draw thumb tip/IP info label."""
    y_pos = 150 if hand_name == "Left" else 180
    cv2.putText(
        frame,
        f"{hand_name} TipY:{thumb_tip.y:.3f} IPY:{thumb_ip.y:.3f}",
        (10, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
    )


# ── HUD: Gesture ──────────────────────────────────────────────────────────────

GEST_COLORS = {
    'FIST':    (0, 255,  80),
    'OPEN':    (0,  80, 255),
    'NEUTRAL': (0, 220,  80),
}
GEST_LABELS = {
    'FIST':    '[FIST]  ACCELERATE',
    'OPEN':    '[OPEN]  BRAKE',
    'NEUTRAL': '[AUTO]  ACCELERATE',
}

def draw_gesture_hud(frame, current_gesture):
    """Draw the current gesture label."""
    cv2.putText(
        frame, GEST_LABELS[current_gesture],
        (10, 210),
        cv2.FONT_HERSHEY_DUPLEX, 0.65,
        GEST_COLORS[current_gesture], 1, cv2.LINE_AA
    )


# ── HUD: Pedal Bar ────────────────────────────────────────────────────────────

def draw_pedal_bar(frame, rt_val, lt_val, h, w):
    """Draw the vertical throttle/brake bar on the right side."""
    bar_top    = 60
    bar_bottom = h - 60
    bar_height = bar_bottom - bar_top
    bar_x      = w - 30

    cv2.rectangle(frame, (bar_x - 10, bar_top),  (bar_x + 10, bar_bottom), (40, 40, 40),   -1)
    cv2.rectangle(frame, (bar_x - 10, bar_top),  (bar_x + 10, bar_bottom), (120, 120, 120), 1)

    if rt_val > 0:
        fill_h = int((rt_val / 255.0) * (bar_height // 2))
        cv2.rectangle(frame,
                      (bar_x - 8, bar_top + bar_height // 2 - fill_h),
                      (bar_x + 8, bar_top + bar_height // 2),
                      (0, 220, 80), -1)

    if lt_val > 0:
        fill_h = int((lt_val / 255.0) * (bar_height // 2))
        cv2.rectangle(frame,
                      (bar_x - 8, bar_top + bar_height // 2),
                      (bar_x + 8, bar_top + bar_height // 2 + fill_h),
                      (0, 80, 220), -1)

    cv2.putText(frame, 'GAS', (bar_x - 14, bar_top - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 80), 1)
    cv2.putText(frame, 'BRK', (bar_x - 14, bar_bottom + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 80, 220), 1)
