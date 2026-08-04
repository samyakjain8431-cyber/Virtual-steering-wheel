import pydirectinput
import time


print("Pressing A")
pydirectinput.keyDown("a")
time.sleep(2)
pydirectinput.keyUp("a")

print("Done")

