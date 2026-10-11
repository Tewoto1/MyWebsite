"""Rubik's cube model: 54 stickers as 3D quads, plus one slice that can be turned by hand.

Coordinates: cube centred at the origin, cubies at -1/0/1, stickers on the |x|=1.5 faces.
Face ids: 0 R(+x) 1 L(-x) 2 U(+y) 3 D(-y) 4 F(+z) 5 B(-z).
A slice is (axis, layer): the cubies whose coordinate along `axis` is `layer`.
Angles are counter-clockwise about the +axis.
"""
import itertools
import numpy as np

from easing import smooth

AXES = np.eye(3)
QUARTER = np.pi / 2


def rotation(axis, angle):
    """Rotation matrix about a unit axis (Rodrigues' formula)."""
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def axis_angle(R):
    """Unit axis and angle of a rotation matrix (the inverse of rotation())."""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.linalg.norm(axis) > 1e-9:
        return axis / np.linalg.norm(axis), angle
    if angle < 1e-6:  # no rotation at all
        return AXES[0], 0.0
    M = (R + np.eye(3)) / 2  # exactly half a turn: the axis is any nonzero column of (R + I) / 2
    k = np.argmax(np.diag(M))
    return M[:, k] / np.sqrt(M[k, k]), angle


def square(axis, offset, half):
    """Corners of a square perpendicular to `axis`, at `offset` along it."""
    u, v = AXES[(axis + 1) % 3], AXES[(axis + 2) % 3]
    return np.array([offset * AXES[axis] + half * (su * u + sv * v) for su, sv in ((1, 1), (-1, 1), (-1, -1), (1, -1))])


def solved():
    """Corners (54, 4, 3), normals (54, 3) and face ids (54,) of a solved cube."""
    corners, normals, faces = [], [], []
    for face, (axis, sign) in enumerate(itertools.product(range(3), (1, -1))):
        u, v = AXES[(axis + 1) % 3], AXES[(axis + 2) % 3]
        for i, j in itertools.product((-1, 0, 1), repeat=2):
            corners.append(square(axis, 1.5 * sign, 0.5) + i * u + j * v)
            normals.append(sign * AXES[axis])
            faces.append(face)
    return np.array(corners), np.array(normals), np.array(faces)


class RubiksCube:
    def __init__(self, settle_time=0.3, curve=smooth):
        self.settle_time, self.curve = settle_time, curve
        self.reset()

    def reset(self):
        self.corners, self.normals, self.faces = solved()
        self.turn = None  # (axis, layer) of the slice being turned
        self.moving = None  # mask of the stickers in it
        self.angle = 0.0  # how far it is turned
        self.settling = None  # (from_angle, to_angle, start_time) once released

    @property
    def held(self):
        return self.turn is not None and self.settling is None

    def grab(self, axis, layer):
        """Start turning a slice by hand, first settling any previous turn."""
        self.finish()
        self.turn, self.angle = (axis, layer), 0.0
        self.moving = self._slice(axis, layer)

    def release(self, t, snap=True):
        """Let go at time t: ease to the nearest quarter turn, or back to 0 if not snap."""
        target = np.round(self.angle / QUARTER) * QUARTER if snap else 0.0
        self.settling = (self.angle, target, t)

    def finish(self):
        """Jump the turn to where it is heading and bake it into the stickers."""
        if self.turn:
            angle = self.settling[1] if self.settling else self.angle
            self._apply(*self.turn, round(angle / QUARTER))
            self.turn = self.settling = None
            self.angle = 0.0

    def update(self, t):
        """Advance any settling animation to time t; return (corners, normals) of all quads.

        The first 54 quads are the stickers. While a slice is turned, extra quads follow:
        the bare insides of the cube exposed along the cut(s)."""
        if self.settling:
            a0, a1, t0 = self.settling
            p = min((t - t0) / self.settle_time, 1.0)
            self.angle = a0 + (a1 - a0) * self.curve(p)
            if p >= 1:
                self.finish()
        if not self.turn:
            return self.corners, self.normals
        axis, layer = self.turn
        R = rotation(AXES[axis], self.angle)
        corners, normals = self.corners.copy(), self.normals.copy()
        corners[self.moving] = corners[self.moving] @ R.T
        normals[self.moving] = normals[self.moving] @ R.T
        cuts, cut_normals = [], []
        for side in (1, -1):
            offset = layer + 0.5 * side
            if abs(offset) < 1.5:  # there is a neighbouring slice on this side
                n = side * AXES[axis]
                cuts += [square(axis, offset, 1.5) @ R.T, square(axis, offset, 1.5)]
                cut_normals += [n @ R.T, -n]  # turning side faces out, still side faces in
        return np.concatenate([corners, cuts]), np.concatenate([normals, cut_normals])

    def scramble(self, n=25, rng=None):
        rng = rng or np.random.default_rng()
        self.finish()
        for _ in range(n):
            self._apply(rng.integers(3), rng.integers(-1, 2), rng.choice((-1, 1)))

    def slice_move(self, i, direction):
        """(axis, layer, sign): turning that slice by +sign pushes sticker i along `direction`,
        an axis lying in its face."""
        a = np.round(np.cross(self.normals[i], direction))
        axis = np.abs(a).argmax()
        return axis, int(np.round(self._cubies()[i, axis])), int(a[axis])

    def _cubies(self):
        """Centre of the cubie each sticker belongs to."""
        return self.corners.mean(axis=1) - 0.5 * self.normals

    def _slice(self, axis, layer):
        return np.round(self._cubies()[:, axis]) == layer

    def _apply(self, axis, layer, quarters):
        """Turn a slice by whole quarter turns, with an exact integer matrix so floats never drift."""
        mask = self._slice(axis, layer)
        R = np.round(rotation(AXES[axis], quarters * QUARTER))
        self.corners[mask] = self.corners[mask] @ R.T
        self.normals[mask] = self.normals[mask] @ R.T
