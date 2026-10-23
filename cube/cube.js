// Rubik's cube model: 54 stickers as 3D quads, plus one slice that can be turned by hand.
//
// Coordinates: cube centred at the origin, cubies at -1/0/1, stickers on the |x|=1.5 faces.
// Face ids: 0 R(+x) 1 L(-x) 2 U(+y) 3 D(-y) 4 F(+z) 5 B(-z).
// A slice is (axis, layer): the cubies whose coordinate along `axis` is `layer`.
// Angles are counter-clockwise about the +axis.
//
// Vectors are [x, y, z]; matrices are row-major [[...], [...], [...]].
// numpy's `points @ R.T` is `apply(R, p)` here.

import { smooth } from "./easing.js";

export const AXES = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
export const QUARTER = Math.PI / 2;

// ---- Small vector / matrix helpers ----

export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm = (a) => Math.hypot(a[0], a[1], a[2]);
export const apply = (R, p) => [dot(R[0], p), dot(R[1], p), dot(R[2], p)];
export const transpose = (A) => [0, 1, 2].map((j) => [A[0][j], A[1][j], A[2][j]]);
export const matmul = (A, B) => {
  const Bt = transpose(B);
  return A.map((row) => Bt.map((col) => dot(row, col)));
};
const mean = (quad) => scale(quad.reduce(add), 1 / quad.length);

// Rotation matrix about a unit axis (Rodrigues' formula).
export function rotation([x, y, z], angle) {
  const K = [[0, -z, y], [z, 0, -x], [-y, x, 0]];
  const KK = matmul(K, K);
  const s = Math.sin(angle), c = 1 - Math.cos(angle);
  return [0, 1, 2].map((i) => [0, 1, 2].map((j) => (i === j ? 1 : 0) + s * K[i][j] + c * KK[i][j]));
}

// Unit axis and angle of a rotation matrix (the inverse of rotation()).
export function axisAngle(R) {
  const trace = R[0][0] + R[1][1] + R[2][2];
  const angle = Math.acos(Math.min(Math.max((trace - 1) / 2, -1), 1));
  const axis = [R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]];
  if (norm(axis) > 1e-9) return [scale(axis, 1 / norm(axis)), angle];
  if (angle < 1e-6) return [AXES[0], 0]; // no rotation at all
  // Exactly half a turn: the axis is any nonzero column of (R + I) / 2
  const M = R.map((row, i) => row.map((v, j) => (v + (i === j ? 1 : 0)) / 2));
  let k = 0;
  for (let d = 1; d < 3; d++) if (M[d][d] > M[k][k]) k = d;
  return [[M[0][k], M[1][k], M[2][k]].map((v) => v / Math.sqrt(M[k][k])), angle];
}

// Corners of a square perpendicular to `axis`, at `offset` along it.
export function square(axis, offset, half) {
  const u = AXES[(axis + 1) % 3], v = AXES[(axis + 2) % 3];
  return [[1, 1], [-1, 1], [-1, -1], [1, -1]].map(([su, sv]) =>
    add(scale(AXES[axis], offset), scale(add(scale(u, su), scale(v, sv)), half)));
}

// Corners (54 x 4 x 3), normals (54 x 3) and face ids (54) of a solved cube.
function solved() {
  const corners = [], normals = [], faces = [];
  let face = 0;
  for (let axis = 0; axis < 3; axis++) {
    for (const sign of [1, -1]) {
      const u = AXES[(axis + 1) % 3], v = AXES[(axis + 2) % 3];
      for (const i of [-1, 0, 1]) {
        for (const j of [-1, 0, 1]) {
          const shift = add(scale(u, i), scale(v, j));
          corners.push(square(axis, 1.5 * sign, 0.5).map((p) => add(p, shift)));
          normals.push(scale(AXES[axis], sign));
          faces.push(face);
        }
      }
      face++;
    }
  }
  return [corners, normals, faces];
}

export class RubiksCube {
  constructor(settleTime = 0.3, curve = smooth) {
    this.settleTime = settleTime;
    this.curve = curve;
    this.reset();
  }

  reset() {
    [this.corners, this.normals, this.faces] = solved();
    this.turn = null; // [axis, layer] of the slice being turned
    this.moving = null; // mask of the stickers in it
    this.angle = 0; // how far it is turned
    this.settling = null; // [fromAngle, toAngle, startTime] once released
  }

  get held() {
    return this.turn !== null && this.settling === null;
  }

  // Start turning a slice by hand, first settling any previous turn.
  grab(axis, layer) {
    this.finish();
    this.turn = [axis, layer];
    this.angle = 0;
    this.moving = this._slice(axis, layer);
  }

  // Let go at time t: ease to the nearest quarter turn, or back to 0 if not snap.
  release(t, snap = true) {
    const target = snap ? Math.round(this.angle / QUARTER) * QUARTER : 0;
    this.settling = [this.angle, target, t];
  }

  // Jump the turn to where it is heading and bake it into the stickers.
  finish() {
    if (this.turn) {
      const angle = this.settling ? this.settling[1] : this.angle;
      this._apply(...this.turn, Math.round(angle / QUARTER));
      this.turn = this.settling = null;
      this.angle = 0;
    }
  }

  // Advance any settling animation to time t; return [corners, normals] of all quads.
  // The first 54 quads are the stickers. While a slice is turned, extra quads follow:
  // the bare insides of the cube exposed along the cut(s).
  update(t) {
    if (this.settling) {
      const [a0, a1, t0] = this.settling;
      const p = Math.min((t - t0) / this.settleTime, 1);
      this.angle = a0 + (a1 - a0) * this.curve(p);
      if (p >= 1) this.finish();
    }
    if (!this.turn) return [this.corners, this.normals];
    const [axis, layer] = this.turn;
    const R = rotation(AXES[axis], this.angle);
    const corners = this.corners.map((q, i) => (this.moving[i] ? q.map((p) => apply(R, p)) : q));
    const normals = this.normals.map((n, i) => (this.moving[i] ? apply(R, n) : n));
    for (const side of [1, -1]) {
      const offset = layer + 0.5 * side;
      if (Math.abs(offset) < 1.5) { // there is a neighbouring slice on this side
        const n = scale(AXES[axis], side);
        const sq = square(axis, offset, 1.5);
        corners.push(sq.map((p) => apply(R, p)), sq);
        normals.push(apply(R, n), scale(n, -1)); // turning side faces out, still side faces in
      }
    }
    return [corners, normals];
  }

  scramble(n = 25) {
    const randint = (lo, hi) => lo + Math.floor(Math.random() * (hi - lo)); // [lo, hi)
    this.finish();
    for (let k = 0; k < n; k++) this._apply(randint(0, 3), randint(-1, 2), randint(0, 2) ? 1 : -1);
  }

  // [axis, layer, sign]: turning that slice by +sign pushes sticker i along `direction`,
  // an axis lying in its face.
  sliceMove(i, direction) {
    const a = cross(this.normals[i], direction).map(Math.round);
    let axis = 0;
    for (let d = 1; d < 3; d++) if (Math.abs(a[d]) > Math.abs(a[axis])) axis = d;
    return [axis, Math.round(this._cubies()[i][axis]), a[axis]];
  }

  // Centre of the cubie each sticker belongs to.
  _cubies() {
    return this.corners.map((q, i) => sub(mean(q), scale(this.normals[i], 0.5)));
  }

  _slice(axis, layer) {
    return this._cubies().map((c) => Math.round(c[axis]) === layer);
  }

  // Turn a slice by whole quarter turns, with an exact integer matrix so floats never drift.
  _apply(axis, layer, quarters) {
    const mask = this._slice(axis, layer);
    const R = rotation(AXES[axis], quarters * QUARTER).map((row) => row.map((v) => Math.round(v) + 0)); // +0 drops -0
    this.corners = this.corners.map((q, i) => (mask[i] ? q.map((p) => apply(R, p)) : q));
    this.normals = this.normals.map((n, i) => (mask[i] ? apply(R, n) : n));
  }
}

export { mean };
