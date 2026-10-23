import * as easing from "./easing.js";
import { AXES, RubiksCube, add, apply, axisAngle, matmul, mean, rotation, transpose } from "./cube.js";
import { Screen, facing, seams } from "./raster.js";

const canvas = document.getElementById("cube");
const ctx = canvas.getContext("2d");

// Display choices
const FONT_PX = 10;
const LETTERS = "abcdef"; // face letters, by face id
const NOISE = "ghijklmnopqrstuvwxyz"; // random letters for everything that isn't a face
const FACE_INK = "#333", NOISE_INK = "#aaa";
const SIZE = 0.5; // cube width as a fraction of the smaller screen side
const CHANGE_RATE = 0.5; // random changes per letter per second
const GLITCH_MS = 250; // how long a changed face letter stays random before resetting

// Feel
const SETTLE_TIME = 0.35; // seconds for a released slice to ease into place
const SETTLE_CURVE = easing.outBack; // any function in easing.js
const SNAP = true; // true: settle to the nearest quarter turn; false: always spring back to where it started
const ORBIT = 0.008; // radians of view rotation per dragged pixel
const DRAG_PX = 6; // movement before a press on a sticker picks a slice to turn
const HOME = matmul(rotation([1, 0, 0], Math.atan(1 / Math.sqrt(2))), rotation([0, 1, 0], -Math.PI / 4)); // isometric: edges form a Y
const HOME_TIME = 0.6; // seconds to swing back home on double-click
const HOME_CURVE = easing.smooth;

// Global state
let screen = null;
let noise = null; // random letter per text cell
let glitch = null; // per text cell: time (ms) until which a face letter shows its random letter instead
let grid = null; // quad id per text cell, from the last frame
const cube = new RubiksCube(SETTLE_TIME, SETTLE_CURVE);
let view = HOME;
let homing = null; // [startView, axis, angle, startTime] while swinging back home
let now = 0; // seconds, from the last animation frame
const presses = new Map(); // pointerId -> {start, last, sticker, turn}

const randomLetter = () => NOISE[Math.floor(Math.random() * NOISE.length)];

function setup() {
  const dpr = window.devicePixelRatio;
  const W = window.innerWidth, H = window.innerHeight;
  canvas.width = Math.floor(W * dpr);
  canvas.height = Math.floor(H * dpr);
  // Resizing resets the context, so configure it afterwards
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.font = `${FONT_PX}px monospace`;
  ctx.textBaseline = "top";
  screen = new Screen(W, H, [ctx.measureText("M").width, FONT_PX * 1.2], SIZE);
  noise = Array.from({ length: screen.rows * screen.cols }, randomLetter);
  glitch = new Float64Array(screen.rows * screen.cols);
  grid = null;
}

// ---- Rendering ----

function frame(ms) {
  const dt = ms / 1000 - now;
  now = ms / 1000;
  shuffle(ms, dt);
  swingHome(now);
  let [corners, normals] = cube.update(now);
  corners = corners.map((q) => q.map((p) => apply(view, p)));
  normals = normals.map((n) => apply(view, n));
  const ids = facing(corners, normals);
  grid = screen.paint(ids.map((i) => corners[i]), ids);
  draw(grid, ms);
  window.requestAnimationFrame(frame);
}

// Randomly change letters; face letters that change stay random for GLITCH_MS.
function shuffle(ms, dt) {
  const p = CHANGE_RATE * dt;
  for (let k = 0; k < noise.length; k++) {
    if (Math.random() < p) {
      noise[k] = randomLetter();
      glitch[k] = ms + GLITCH_MS;
    }
  }
}

function swingHome(t) {
  if (!homing) return;
  const [startView, axis, angle, t0] = homing;
  const p = Math.min((t - t0) / HOME_TIME, 1);
  view = matmul(rotation(axis, angle * HOME_CURVE(p)), startView);
  if (p >= 1) {
    view = HOME;
    homing = null;
  }
}

// Face cells show their face letter (random while glitched); the rest of the cube,
// i.e. gaps between stickers and insides exposed by a turn, shows random letters.
function draw(grid, ms) {
  const { rows, cols, ch } = screen;
  const nFaces = cube.faces.length;
  const seam = seams(grid, rows, cols);
  ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
  const faceRows = [], noiseRows = [];
  for (let r = 0; r < rows; r++) {
    let face = "", rest = "", anyFace = false, anyRest = false;
    for (let c = 0; c < cols; c++) {
      const k = r * cols + c, g = grid[k];
      if (g < 0) {
        face += " ";
        rest += " ";
      } else if (g < nFaces && !seam[k]) {
        face += glitch[k] > ms ? noise[k] : LETTERS[cube.faces[g]];
        rest += " ";
        anyFace = true;
      } else {
        face += " ";
        rest += noise[k];
        anyRest = true;
      }
    }
    if (anyFace) faceRows.push([face, r]);
    if (anyRest) noiseRows.push([rest, r]);
  }
  ctx.fillStyle = FACE_INK;
  for (const [text, r] of faceRows) ctx.fillText(text, 0, r * ch);
  ctx.fillStyle = NOISE_INK;
  for (const [text, r] of noiseRows) ctx.fillText(text, 0, r * ch);
}

// ---- Input ----

const sub2 = (a, b) => [a[0] - b[0], a[1] - b[1]];
const dot2 = (a, b) => a[0] * b[0] + a[1] * b[1];

// Sticker drawn under pixel (x, y), or -1.
function stickerAt(x, y) {
  const [r, c] = screen.cell(x, y);
  if (grid === null || !(r >= 0 && r < screen.rows && c >= 0 && c < screen.cols)) return -1;
  const g = grid[r * screen.cols + c];
  return g < cube.faces.length ? g : -1;
}

// Start turning the slice that best follows dragging sticker i along screen vector `drag`.
// Returns the radians-per-pixel vector: slice angle = (pointer - start) . it.
function grab(i, drag) {
  cube.finish();
  const mid = mean(cube.corners[i]);
  let best = null;
  for (const axis of AXES.filter((_, d) => cube.normals[i][d] === 0)) { // the two directions the sticker can slide
    const a = screen.project(apply(view, mid)), b = screen.project(apply(view, add(mid, axis)));
    const step = sub2(b, a);
    const along = dot2(step, drag) / Math.hypot(...step);
    if (best === null || Math.abs(along) > Math.abs(best[0])) best = [along, axis, step];
  }
  const [, axis, step] = best;
  const [turnAxis, layer, sign] = cube.sliceMove(i, axis);
  cube.grab(turnAxis, layer);
  // One cubie of travel on screen spans `step` pixels; the face sits 1.5 cubies from the turn axis
  const s = sign / dot2(step, step) / 1.5;
  return [step[0] * s, step[1] * s];
}

function onDown(e) {
  const pos = [e.clientX, e.clientY];
  const sticker = cube.held ? -1 : stickerAt(...pos); // only one slice turns at a time
  presses.set(e.pointerId, { start: pos, last: pos, sticker, turn: null });
  e.target.setPointerCapture(e.pointerId);
}

function onMove(e) {
  const press = presses.get(e.pointerId);
  if (!press) return;
  const pos = [e.clientX, e.clientY];
  if (press.sticker < 0) { // background drag: orbit the whole cube
    const dx = (pos[0] - press.last[0]) * ORBIT, dy = (pos[1] - press.last[1]) * ORBIT;
    homing = null;
    view = matmul(matmul(rotation([0, 1, 0], dx), rotation([1, 0, 0], dy)), view);
  } else if (press.turn === null) {
    const d = sub2(pos, press.start);
    if (Math.hypot(...d) > DRAG_PX) press.turn = grab(press.sticker, d);
  } else {
    cube.angle = dot2(sub2(pos, press.start), press.turn);
  }
  press.last = pos;
}

function onUp(e) {
  const press = presses.get(e.pointerId);
  presses.delete(e.pointerId);
  if (press && press.turn !== null) cube.release(now, SNAP);
}

function onDoubleClick() {
  homing = [view, ...axisAngle(matmul(HOME, transpose(view))), now];
}

function onKey(e) {
  if (e.key === "s") cube.scramble();
  else if (e.key === "r") cube.reset();
}

setup();
for (const [event, handler] of [["resize", setup], ["pointerdown", onDown], ["pointermove", onMove],
  ["pointerup", onUp], ["pointercancel", onUp], ["dblclick", onDoubleClick], ["keydown", onKey]]) {
  window.addEventListener(event, handler);
}
window.requestAnimationFrame(frame);
