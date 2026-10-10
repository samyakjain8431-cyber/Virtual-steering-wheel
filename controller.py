import vgamepad as vg
from pynput.keyboard import Controller as KeyboardController


# ── Virtual Xbox Controller ────────────────────────────────────────────────────

gamepad = vg.VX360Gamepad()
gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
gamepad.update()


# ── Keyboard controller for W / S key simulation ──────────────────────────────

kb = KeyboardController()

# Track which keys are currently held so we don't spam press/release
_key_held = {'w': False, 's': False}


def set_key(key_char: str, pressed: bool):
    """Press or release a single key only when its state actually changes."""
    if _key_held[key_char] == pressed:
        return
    if pressed:
        kb.press(key_char)
    else:
        kb.release(key_char)
    _key_held[key_char] = pressed


def release_all_keys():
    """Release all held keys — call on exit."""
    set_key('w', False)
    set_key('s', False)


def reset_gamepad():
    """Centre stick and zero triggers — call on exit."""
    gamepad.right_trigger(value=0)
    gamepad.left_trigger(value=0)
    gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
    gamepad.update()
