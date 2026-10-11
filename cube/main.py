import numpy as np
from js import document, window
from pyodide.ffi import create_proxy

import easing
from cube import AXES, RubiksCube, axis_angle, rotation
from raster import Screen, facing, seams

canvas = document.getElementById("cube")
ctx = canvas.getContext("2d")
rng = np.random.default_rng()

# Display choices
FONT_PX = 10
LETTERS = np.array(list("abcdef"))  # face letters, by face id
NOISE = np.array(list("ghijklmnopqrstuvwxyz"))  # random letters for everything that isn't a face
FACE_INK, NOISE_INK = "#333", "#aaa"
SIZE = 0.5  # cube width as a fraction of the smaller screen side
CHANGE_RATE = 0.5  # random changes per letter per second
GLITCH_MS = 250  # how long a changed face letter stays random before resetting

# Feel
SETTLE_TIME = 0.35  # seconds for a released slice to ease into place
SETTLE_CURVE = easing.out_back  # any function in easing.py
SNAP = True  # True: settle to the nearest quarter turn; False: always spring back to where it started
ORBIT = 0.008  # radians of view rotation per dragged pixel
DRAG_PX = 6  # movement before a press on a sticker picks a slice to turn
HOME = rotation([1, 0, 0], np.arctan(1 / np.sqrt(2))) @ rotation([0, 1, 0], -np.pi / 4)  # isometric: edges form a Y
HOME_TIME = 0.6  # seconds to swing back home on double-click
HOME_CURVE = easing.smooth

# Global state
screen = None
noise = None  # random letter per text cell
glitch = None  # per text cell: time (ms) until which a face letter shows its random letter instead
grid = None  # quad id per text cell, from the last frame
cube = RubiksCube(SETTLE_TIME, SETTLE_CURVE)
view = HOME
homing = None  # (start_view, axis, angle, start_time) while swinging back home
now = 0.0  # seconds, from the last animation frame
presses = {}  # pointerId -> {"start", "last", "sticker", "turn"}


def setup(*_):
    global screen, noise, glitch
    dpr = window.devicePixelRatio
    W, H = window.innerWidth, window.innerHeight
    canvas.width, canvas.height = int(W * dpr), int(H * dpr)
    # Resizing resets the context, so configure it afterwards
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.font = f"{FONT_PX}px monospace"
    ctx.textBaseline = "top"
    screen = Screen(W, H, (ctx.measureText("M").width, FONT_PX * 1.2), SIZE)
    noise = rng.choice(NOISE, size=screen.shape)
    glitch = np.zeros(screen.shape)


# ---- Rendering ----

def frame(ms):
    global now, grid
    dt, now = ms / 1000 - now, ms / 1000
    shuffle(ms, dt)
    swing_home(now)
    corners, normals = cube.update(now)
    corners, normals = corners @ view.T, normals @ view.T
    ids = facing(corners, normals)
    grid = screen.paint(corners[ids], ids)
    draw(grid, ms)
    window.requestAnimationFrame(frame_proxy)


def shuffle(ms, dt):
    """Randomly change letters; face letters that change stay random for GLITCH_MS."""
    flip = rng.random(noise.shape) < CHANGE_RATE * dt
    noise[flip] = rng.choice(NOISE, size=flip.sum())
    glitch[flip] = ms + GLITCH_MS


def swing_home(t):
    global view, homing
    if homing:
        start_view, axis, angle, t0 = homing
        p = min((t - t0) / HOME_TIME, 1.0)
        view = rotation(axis, angle * HOME_CURVE(p)) @ start_view
        if p >= 1:
            view, homing = HOME, None


def draw(grid, ms):
    """Face cells show their face letter (random while glitched); the rest of the cube,
    i.e. gaps between stickers and insides exposed by a turn, shows random letters."""
    on_cube = grid >= 0
    face = on_cube & (grid < len(cube.faces)) & ~seams(grid)
    letters = LETTERS[cube.faces[np.clip(grid, 0, len(cube.faces) - 1)]]
    face_text = np.where(glitch > ms, noise, letters)
    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight)
    for mask, text, ink in ((face, face_text, FACE_INK), (on_cube & ~face, noise, NOISE_INK)):
        ctx.fillStyle = ink
        layer = np.where(mask, text, " ")
        for r in np.flatnonzero(mask.any(axis=1)):
            ctx.fillText("".join(layer[r]), 0, r * screen.ch)


# ---- Input ----

def sticker_at(x, y):
    """Sticker drawn under pixel (x, y), or -1."""
    r, c = screen.cell(x, y)
    if grid is None or not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
        return -1
    return grid[r, c] if grid[r, c] < len(cube.faces) else -1


def grab(i, drag):
    """Start turning the slice that best follows dragging sticker i along screen vector `drag`.
    Returns the radians-per-pixel vector: slice angle = (pointer - start) @ it."""
    cube.finish()
    mid = cube.corners[i].mean(axis=0)
    best = None
    for axis in AXES[cube.normals[i] == 0]:  # the two directions the sticker can slide
        a, b = screen.project(np.array([mid, mid + axis]) @ view.T)
        along = (b - a) @ drag / np.linalg.norm(b - a)
        if best is None or abs(along) > abs(best[0]):
            best = along, axis, b - a
    along, axis, step = best
    turn_axis, layer, sign = cube.slice_move(i, axis)
    cube.grab(turn_axis, layer)
    # One cubie of travel on screen spans `step` pixels; the face sits 1.5 cubies from the turn axis
    return sign * step / (step @ step) / 1.5


def on_down(e):
    pos = np.array([e.clientX, e.clientY])
    sticker = -1 if cube.held else sticker_at(*pos)  # only one slice turns at a time
    presses[e.pointerId] = {"start": pos, "last": pos, "sticker": sticker, "turn": None}
    e.target.setPointerCapture(e.pointerId)


def on_move(e):
    global view, homing
    press = presses.get(e.pointerId)
    if press is None:
        return
    pos = np.array([e.clientX, e.clientY])
    if press["sticker"] < 0:  # background drag: orbit the whole cube
        dx, dy = (pos - press["last"]) * ORBIT
        homing = None
        view = rotation([0, 1, 0], dx) @ rotation([1, 0, 0], dy) @ view
    elif press["turn"] is None:
        if np.linalg.norm(pos - press["start"]) > DRAG_PX:
            press["turn"] = grab(press["sticker"], pos - press["start"])
    else:
        cube.angle = (pos - press["start"]) @ press["turn"]
    press["last"] = pos


def on_up(e):
    press = presses.pop(e.pointerId, None)
    if press and press["turn"] is not None:
        cube.release(now, SNAP)


def on_double_click(e):
    global homing
    homing = (view, *axis_angle(HOME @ view.T), now)


def on_key(e):
    if e.key == "s":
        cube.scramble()
    elif e.key == "r":
        cube.reset()


frame_proxy = create_proxy(frame)
setup()
for event, handler in (("resize", setup), ("pointerdown", on_down), ("pointermove", on_move),
                       ("pointerup", on_up), ("pointercancel", on_up), ("dblclick", on_double_click), ("keydown", on_key)):
    window.addEventListener(event, create_proxy(handler))
window.requestAnimationFrame(frame_proxy)
