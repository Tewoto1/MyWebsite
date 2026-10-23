// View-space quads -> rows x cols grid of quad ids, one per text cell (-1 = empty).
//
// The camera sits at (0, 0, CAMERA) looking down -z; screen y points down.
// Grids are flat typed arrays indexed by r * cols + c.

import { cross, dot, mean, sub } from "./cube.js";

export const CAMERA = 9.0; // eye distance from the cube centre, in cubie units

// Indices of quads whose front side faces the camera.
export function facing(corners, normals) {
  const ids = [];
  corners.forEach((q, i) => {
    if (dot(normals[i], sub([0, 0, CAMERA], mean(q))) > 0) ids.push(i);
  });
  return ids;
}

// Is point (x, y) inside a convex polygon of either winding?
function inside(poly, x, y) {
  let pos = true, neg = true;
  for (let k = 0; k < poly.length; k++) {
    const [ax, ay] = poly[k], [bx, by] = poly[(k + 1) % poly.length];
    const c = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
    pos &&= c >= 0;
    neg &&= c <= 0;
  }
  return pos || neg;
}

// Cells where one quad meets another, left blank so neighbouring stickers read apart.
export function seams(grid, rows, cols) {
  const seam = new Uint8Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const k = r * cols + c, g = grid[k];
      if (c + 1 < cols && g !== grid[k + 1] && grid[k + 1] >= 0) seam[k] = 1;
      if (r + 1 < rows && g !== grid[k + cols] && grid[k + cols] >= 0) seam[k] = 1;
    }
  }
  return seam;
}

export class Screen {
  // size: cube width as a fraction of the smaller screen side.
  constructor(width, height, [cw, ch], size) {
    this.cw = cw;
    this.ch = ch;
    this.rows = Math.floor(height / ch) + 1;
    this.cols = Math.floor(width / cw) + 1;
    this.scale = (size * Math.min(width, height)) / 3;
    this.center = [width / 2, height / 2];
  }

  // Perspective-project a 3D point to [x, y] pixels.
  project(p) {
    const f = (this.scale * CAMERA) / (CAMERA - p[2]);
    return [this.center[0] + p[0] * f, this.center[1] - p[1] * f];
  }

  // [row, col] of the text cell under pixel (x, y).
  cell(x, y) {
    return [Math.floor(y / this.ch), Math.floor(x / this.cw)];
  }

  // Z-buffered fill: each cell gets the id of the nearest quad covering its centre.
  paint(quads, ids) {
    const { rows, cols, cw, ch } = this;
    const grid = new Int32Array(rows * cols).fill(-1);
    const zbuf = new Float64Array(rows * cols).fill(Infinity);
    quads.forEach((quad, n) => {
      const flat = quad.map((p) => this.project(p));
      const xs = flat.map((p) => p[0]), ys = flat.map((p) => p[1]);
      const c0 = Math.max(Math.floor(Math.min(...xs) / cw), 0), c1 = Math.min(Math.floor(Math.max(...xs) / cw) + 1, cols);
      const r0 = Math.max(Math.floor(Math.min(...ys) / ch), 0), r1 = Math.min(Math.floor(Math.max(...ys) / ch) + 1, rows);
      if (!(c0 < c1 && r0 < r1)) return;
      const depth = this.depth(quad);
      for (let r = r0; r < r1; r++) {
        const y = (r + 0.5) * ch;
        for (let c = c0; c < c1; c++) {
          const x = (c + 0.5) * cw, k = r * cols + c;
          const z = depth(x, y);
          if (z < zbuf[k] && inside(flat, x, y)) {
            grid[k] = ids[n];
            zbuf[k] = z;
          }
        }
      }
    });
    return grid;
  }

  // Function (x, y) -> distance in front of the eye of the quad's plane, seen through pixel (x, y).
  depth(quad) {
    const n = cross(sub(quad[1], quad[0]), sub(quad[3], quad[0]));
    const num = dot(n, quad[0]) - n[2] * CAMERA;
    const k = this.scale * CAMERA, [cx, cy] = this.center;
    return (x, y) => num / ((n[0] * (x - cx)) / k + (n[1] * (cy - y)) / k - n[2]);
  }
}
