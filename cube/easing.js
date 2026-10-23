// Easing curves: map progress 0..1 to 0..1 (some overshoot on the way).

export const linear = (p) => p;

export const smooth = (p) => p * p * (3 - 2 * p);

export const outCubic = (p) => 1 - (1 - p) ** 3;

// Overshoots slightly, then settles.
export const outBack = (p, k = 1.7) => 1 + (k + 1) * (p - 1) ** 3 + k * (p - 1) ** 2;

// Damped wobble around the target.
export const spring = (p) => 1 - Math.exp(-6 * p) * Math.cos(3 * Math.PI * p);
