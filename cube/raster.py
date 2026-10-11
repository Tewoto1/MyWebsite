"""View-space quads -> (rows, cols) grid of quad ids, one per text cell (-1 = empty).

The camera sits at (0, 0, CAMERA) looking down -z; screen y points down.
"""
import numpy as np

CAMERA = 9.0  # eye distance from the cube centre, in cubie units


def facing(corners, normals):
    """Indices of quads whose front side faces the camera."""
    mids = corners.mean(axis=1)
    return np.flatnonzero(np.einsum("qi,qi->q", normals, [0, 0, CAMERA] - mids) > 0)


def inside(poly, x, y):
    """Mask of points (x, y) inside a convex polygon of either winding."""
    pos = neg = True
    for (ax, ay), (bx, by) in zip(poly, np.roll(poly, -1, axis=0)):
        cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
        pos, neg = pos & (cross >= 0), neg & (cross <= 0)
    return pos | neg


def seams(grid):
    """Cells where one quad meets another, left blank so neighbouring stickers read apart."""
    seam = np.zeros(grid.shape, bool)
    seam[:, :-1] |= (grid[:, :-1] != grid[:, 1:]) & (grid[:, 1:] >= 0)
    seam[:-1] |= (grid[:-1] != grid[1:]) & (grid[1:] >= 0)
    return seam


class Screen:
    def __init__(self, width, height, cell, size):
        """size: cube width as a fraction of the smaller screen side."""
        self.cw, self.ch = cell
        self.shape = (int(height // self.ch) + 1, int(width // self.cw) + 1)
        self.scale = size * min(width, height) / 3
        self.center = np.array([width / 2, height / 2])

    def project(self, points):
        """Perspective-project (..., 3) points to (..., 2) pixels."""
        f = self.scale * CAMERA / (CAMERA - points[..., 2])
        return self.center + np.stack([points[..., 0], -points[..., 1]], axis=-1) * f[..., None]

    def cell(self, x, y):
        """(row, col) of the text cell under pixel (x, y)."""
        return int(y // self.ch), int(x // self.cw)

    def paint(self, quads, ids):
        """Z-buffered fill: each cell gets the id of the nearest quad covering its centre."""
        rows, cols = self.shape
        grid = np.full(self.shape, -1)
        zbuf = np.full(self.shape, np.inf)
        for quad, i in zip(quads, ids):
            flat = self.project(quad)
            (x0, y0), (x1, y1) = flat.min(axis=0), flat.max(axis=0)
            c0, c1 = max(int(x0 // self.cw), 0), min(int(x1 // self.cw) + 1, cols)
            r0, r1 = max(int(y0 // self.ch), 0), min(int(y1 // self.ch) + 1, rows)
            if c0 < c1 and r0 < r1:
                x = (np.arange(c0, c1) + 0.5) * self.cw
                y = (np.arange(r0, r1)[:, None] + 0.5) * self.ch
                z = self.depth(quad, x, y)
                hit = inside(flat, x, y) & (z < zbuf[r0:r1, c0:c1])
                grid[r0:r1, c0:c1][hit] = i
                zbuf[r0:r1, c0:c1][hit] = z[hit]
        return grid

    def depth(self, quad, x, y):
        """Distance in front of the eye of the quad's plane, seen through pixels (x, y)."""
        n = np.cross(quad[1] - quad[0], quad[3] - quad[0])
        u = (x - self.center[0]) / (self.scale * CAMERA)
        v = (self.center[1] - y) / (self.scale * CAMERA)
        with np.errstate(divide="ignore"):
            return (n @ quad[0] - n[2] * CAMERA) / (n[0] * u + n[1] * v - n[2])
