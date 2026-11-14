import * as THREE from 'three';
import {
  clamp,
  noise,
  eq,
  upperArmQuat,
  lowerArmQuat,
  type Side,
  type PoseBuffer,
  type FingerCurls,
} from './pose-utils.js';

const _q = new THREE.Quaternion();

interface ArmSpec {
  raise?: number;
  fwd?: number;
  twist?: number;
  flex?: number;
  ftwist?: number;
  hand?: [number, number, number];
  fingers?: FingerCurls;
}

const FINGER_KEYS = ['thumb', 'index', 'middle', 'ring', 'little', 'all'] as const;
const _mergedCurls: FingerCurls = {};

/**
 * Natural anatomical resting cascade:
 * Index: ~20% curl (0.20)
 * Middle: ~35% curl (0.35)
 * Ring: ~50% curl (0.50)
 * Pinky / Little: ~65% curl (0.65)
 * Thumb: ~15% curl with natural anatomical abduction
 */
export const REST_CASCADE: FingerCurls = {
  index: 0.20,
  middle: 0.35,
  ring: 0.50,
  little: 0.65,
  thumb: 0.15,
  rest: 0.25,
};

export const OPEN_CASCADE: FingerCurls = {
  index: 0.08,
  middle: 0.12,
  ring: 0.18,
  little: 0.24,
  thumb: 0.10,
  rest: 0.18,
};

export const WIDE_CASCADE: FingerCurls = {
  all: 0.04,
  thumb: 0.05,
  rest: 0.10,
};

export const FIST_CASCADE: FingerCurls = {
  index: 0.95,
  middle: 0.98,
  ring: 1.0,
  little: 1.0,
  thumb: 0.90,
  all: 0.98,
  rest: 0.30,
};

export const POINT_CASCADE: FingerCurls = {
  index: 0.02,
  thumb: 0.35,
  middle: 0.88,
  ring: 0.96,
  little: 0.96,
  rest: 0.28,
};

export const THINK_CASCADE: FingerCurls = {
  index: 0.12,
  thumb: 0.30,
  middle: 0.55,
  ring: 0.75,
  little: 0.85,
  rest: 0.25,
};

export const CUPPED_CASCADE: FingerCurls = {
  index: 0.22,
  middle: 0.26,
  ring: 0.32,
  little: 0.38,
  thumb: 0.20,
  rest: 0.25,
};

// Backward-compatible exports
export const OPEN = OPEN_CASCADE;
export const FIST = FIST_CASCADE;

/**
 * 5th-order minimum-jerk polynomial: S_5(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
 * Guarantees C^2 continuous transitions: zero velocity and zero acceleration at boundaries.
 */
export function quintic(tau: number): number {
  const t = clamp(tau, 0, 1);
  return t * t * t * (10 + t * (-15 + 6 * t));
}

const arm = (buf: PoseBuffer, side: Side, w: number, spec: ArmSpec) => {
  const { raise = 0.3, fwd = 0.1, twist = 0, flex = 0.2, ftwist = 0, hand, fingers } = spec;
  const prefix = side;
  buf.blend(`${prefix}UpperArm`, upperArmQuat(side, raise, fwd, twist, _q), w);
  buf.blend(`${prefix}LowerArm`, lowerArmQuat(side, flex, ftwist, _q), w);
  if (hand) buf.blend(`${prefix}Hand`, eq(hand[0], hand[1], hand[2], _q), w);

  const targetFingers = fingers || REST_CASCADE;
  const rest = targetFingers.rest ?? 0.25;
  for (let i = 0; i < FINGER_KEYS.length; i++) {
    const k = FINGER_KEYS[i];
    const v = targetFingers[k];
    const baseRest = REST_CASCADE[k] ?? rest;
    if (typeof v === 'number') {
      _mergedCurls[k] = v * w + (1 - w) * baseRest;
    } else {
      _mergedCurls[k] = baseRest;
    }
  }
  _mergedCurls.rest = rest;
  buf.setFingers(side, _mergedCurls);
};

const addAll = (buf: PoseBuffer, w: number, map: Record<string, [number, number, number]>) => {
  for (const bone in map) {
    const coords = map[bone];
    buf.add(bone, coords[0] * w, coords[1] * w, coords[2] * w);
  }
};

export interface GestureDef {
  duration: number;
  expr?: Record<string, number>;
  apply: (buf: PoseBuffer, ctx: { w: number; t: number }) => void;
}

export const GESTURES: Record<string, GestureDef> = {
  wave: {
    duration: 2.6,
    expr: { happy: 0.45, relaxed: 0.2 },
    apply(buf, { w, t }) {
      const flap = Math.sin(t * 11) * 0.28;
      // Clavicle elevation and subtle protraction
      addAll(buf, w, {
        rightShoulder: [0.02, 0.03, -0.07],
        leftShoulder: [0, 0, 0.02],
        head: [0.02, -0.04, -0.09],
        neck: [0.01, -0.02, -0.03],
        spine: [-0.02, 0, -0.04],
        chest: [0.01, 0, -0.02],
      });
      // Raised right arm waving with pronation/supination wrist sweep
      arm(buf, 'right', w, {
        raise: 1.42,
        fwd: 0.22,
        twist: -1.30,
        flex: 1.10 + flap * 0.35,
        hand: [0.06, -0.18, flap * 0.45],
        fingers: OPEN_CASCADE,
      });
      // Left arm neutral resting cascade
      arm(buf, 'left', w * 0.4, {
        raise: 0.28,
        fwd: 0.12,
        twist: 0.05,
        flex: 0.22,
        hand: [0.05, 0, 0],
        fingers: REST_CASCADE,
      });
    },
  },

  explain: {
    duration: 2.8,
    expr: { relaxed: 0.4, happy: 0.2 },
    apply(buf, { w, t }) {
      const pulse = Math.sin(t * 3.4) * 0.12;
      const headNod = Math.sin(t * 4.2) * 0.03;
      addAll(buf, w, {
        leftShoulder: [0.01, 0.03, 0.03],
        rightShoulder: [0.01, -0.03, -0.03],
        chest: [0.05, 0, 0],
        spine: [0.02, 0, 0],
        head: [0.03 + headNod, Math.sin(t * 1.8) * 0.06, 0],
        neck: [0.015, 0, 0],
      });
      // Bilateral open palms supinated with subtle rhythmic conversational pulsation
      arm(buf, 'left', w, {
        raise: 0.52,
        fwd: 0.68,
        twist: 0.55,
        flex: 1.18 + pulse,
        hand: [0.08, 0.22, 0.12],
        fingers: OPEN_CASCADE,
      });
      arm(buf, 'right', w, {
        raise: 0.52,
        fwd: 0.68,
        twist: -0.55,
        flex: 1.18 - pulse,
        hand: [0.08, -0.22, -0.12],
        fingers: OPEN_CASCADE,
      });
    },
  },

  point: {
    duration: 2.2,
    expr: { happy: 0.25 },
    apply(buf, { w }) {
      addAll(buf, w, {
        rightShoulder: [0.03, 0.05, -0.06],
        leftShoulder: [0, 0, 0.02],
        spine: [0.02, -0.06, 0],
        chest: [0.03, -0.04, 0],
        head: [-0.02, -0.12, 0.03],
        neck: [-0.01, -0.06, 0.01],
      });
      // Right arm extended forward-right with precise index pointer and stable wrist
      arm(buf, 'right', w, {
        raise: 0.88,
        fwd: 1.40,
        twist: -0.15,
        flex: 0.32,
        hand: [0.06, -0.15, 0.05],
        fingers: POINT_CASCADE,
      });
      arm(buf, 'left', w * 0.5, {
        raise: 0.26,
        fwd: 0.14,
        flex: 0.24,
        hand: [0.05, 0, 0],
        fingers: REST_CASCADE,
      });
    },
  },

  think: {
    duration: 3.2,
    expr: { relaxed: 0.35 },
    apply(buf, { w, t }) {
      const breath = Math.sin(t * 0.8) * 0.03;
      addAll(buf, w, {
        rightShoulder: [0.02, 0.04, -0.08],
        leftShoulder: [0, 0, 0.03],
        spine: [0.04, 0.04, -0.05],
        chest: [0.02, 0.03, -0.03],
        head: [-0.09, 0.14 + breath, -0.13],
        neck: [-0.03, 0.06, -0.05],
      });
      // Right hand cups chin / cheek with curled cascade fingers
      arm(buf, 'right', w, {
        raise: 0.48,
        fwd: 0.88,
        twist: -0.45,
        flex: 2.38,
        hand: [0.22, -0.38, 0.14],
        fingers: THINK_CASCADE,
      });
      // Left arm crossed loosely in front of torso supporting right elbow
      arm(buf, 'left', w, {
        raise: 0.32,
        fwd: 0.72,
        twist: 0.32,
        flex: 1.55,
        hand: [0.10, 0.15, 0],
        fingers: REST_CASCADE,
      });
    },
  },

  idea: {
    duration: 2.4,
    expr: { happy: 0.7, surprised: 0.35 },
    apply(buf, { w, t }) {
      const bob = Math.sin(t * 8) * 0.04;
      addAll(buf, w, {
        rightShoulder: [-0.03, 0.02, -0.10],
        leftShoulder: [0, 0, 0.02],
        spine: [-0.07, 0, 0.03],
        chest: [-0.05, 0, 0.02],
        head: [-0.11, -0.02, -0.04],
        neck: [-0.05, 0, 0],
      });
      // Right arm shoots up with eureka index finger point
      arm(buf, 'right', w, {
        raise: 1.18,
        fwd: 0.52,
        twist: -0.58,
        flex: 1.62 + bob,
        hand: [-0.08, -0.18, -0.08],
        fingers: POINT_CASCADE,
      });
      arm(buf, 'left', w * 0.5, {
        raise: 0.35,
        fwd: 0.35,
        twist: 0.25,
        flex: 0.65,
        hand: [0, 0.15, 0],
        fingers: OPEN_CASCADE,
      });
    },
  },

  nod: {
    duration: 1.8,
    expr: { happy: 0.3, relaxed: 0.2 },
    apply(buf, { w, t }) {
      // Damped harmonic multi-peak nod
      const n = Math.sin(t * 8.5) * Math.exp(-t * 0.55) * 0.22;
      addAll(buf, w, {
        head: [n, 0, 0],
        neck: [n * 0.55, 0, 0],
        upperChest: [n * 0.20, 0, 0],
        spine: [n * 0.08, 0, 0],
        leftShoulder: [0, 0, n * 0.06],
        rightShoulder: [0, 0, -n * 0.06],
      });
    },
  },

  shake: {
    duration: 1.8,
    expr: { sad: 0.35, relaxed: 0.15 },
    apply(buf, { w, t }) {
      // Damped harmonic head shake with cervical phase lag and spine counter-yaw
      const s = Math.sin(t * 9.5) * Math.exp(-t * 0.5) * 0.28;
      addAll(buf, w, {
        head: [0, s, s * 0.15],
        neck: [0, s * 0.6, 0],
        spine: [0, -s * 0.14, s * 0.06],
        chest: [0, -s * 0.10, 0],
        leftShoulder: [0, 0, -0.04],
        rightShoulder: [0, 0, 0.04],
      });
    },
  },

  praise: {
    duration: 2.6,
    expr: { happy: 1.0 },
    apply(buf, { w, t }) {
      const bounce = Math.abs(Math.sin(t * 6.5));
      buf.offsetPos('hips', 0, bounce * 0.032 * w, 0);
      addAll(buf, w, {
        leftShoulder: [0, 0, 0.18],
        rightShoulder: [0, 0, -0.18],
        chest: [-0.09, 0, 0],
        spine: [-0.05, 0, 0],
        head: [-0.15, 0, 0],
        neck: [-0.06, 0, 0],
      });
      // Both arms raised high with triumphantly splayed open hands
      arm(buf, 'left', w, {
        raise: 2.38,
        fwd: 0.20,
        twist: 0.35,
        flex: 0.42,
        hand: [-0.15, 0.12, 0.12],
        fingers: WIDE_CASCADE,
      });
      arm(buf, 'right', w, {
        raise: 2.38,
        fwd: 0.20,
        twist: -0.35,
        flex: 0.42,
        hand: [-0.15, -0.12, -0.12],
        fingers: WIDE_CASCADE,
      });
    },
  },

  clap: {
    duration: 2.6,
    expr: { happy: 0.65 },
    apply(buf, { w, t }) {
      // Rhythmic applause oscillation (~3.8 Hz)
      const c = (Math.sin(t * 14) * 0.5 + 0.5) * 0.35;
      addAll(buf, w, {
        leftShoulder: [0.05, 0.08, 0.05],
        rightShoulder: [0.05, -0.08, -0.05],
        chest: [0.03 * c, 0, 0],
        head: [0.06 + 0.03 * c, 0, 0],
      });
      // Both hands brought together in front of chest
      arm(buf, 'left', w, {
        raise: 0.55,
        fwd: 1.15,
        twist: 0.45,
        flex: 1.78 - c * 0.28,
        hand: [0.08, 0.24 + c * 0.15, 0.12],
        fingers: CUPPED_CASCADE,
      });
      arm(buf, 'right', w, {
        raise: 0.55,
        fwd: 1.15,
        twist: -0.45,
        flex: 1.78 - c * 0.28,
        hand: [0.08, -0.24 - c * 0.15, -0.12],
        fingers: CUPPED_CASCADE,
      });
    },
  },

  bow: {
    duration: 2.8,
    expr: { relaxed: 0.75 },
    apply(buf, { w }) {
      buf.offsetPos('hips', 0, -0.018 * w, -0.025 * w);
      addAll(buf, w, {
        spine: [0.28, 0, 0],
        chest: [0.18, 0, 0],
        upperChest: [0.10, 0, 0],
        neck: [0.08, 0, 0],
        head: [0.06, 0, 0],
      });
      // Hands resting neatly along thighs
      arm(buf, 'left', w, {
        raise: 0.16,
        fwd: 0.78,
        twist: 0.18,
        flex: 0.88,
        hand: [0.15, 0.10, 0.05],
        fingers: REST_CASCADE,
      });
      arm(buf, 'right', w, {
        raise: 0.16,
        fwd: 0.78,
        twist: -0.18,
        flex: 0.88,
        hand: [0.15, -0.10, -0.05],
        fingers: REST_CASCADE,
      });
    },
  },

  shy: {
    duration: 2.8,
    expr: { happy: 0.35, relaxed: 0.4 },
    apply(buf, { w, t }) {
      const sway = Math.sin(t * 2) * 0.04;
      addAll(buf, w, {
        leftShoulder: [0.03, 0.04, 0.09],
        rightShoulder: [0, 0, 0.03],
        head: [0.11, 0.14, 0.18],
        neck: [0.04, 0.07, 0.09],
        spine: [0.05, -0.03, -0.06],
        chest: [0.03, 0, -0.04],
      });
      arm(buf, 'left', w, {
        raise: 0.32,
        fwd: 0.82,
        twist: 0.22,
        flex: 1.72 + sway,
        hand: [0.12, 0.18, 0.08],
        fingers: { ...REST_CASCADE, index: 0.35 },
      });
      arm(buf, 'right', w, {
        raise: 0.32,
        fwd: 0.82,
        twist: -0.22,
        flex: 1.72 - sway,
        hand: [0.12, -0.18, -0.08],
        fingers: { ...REST_CASCADE, index: 0.35 },
      });
    },
  },

  encourage: {
    duration: 2.4,
    expr: { happy: 0.65 },
    apply(buf, { w, t }) {
      const pump = Math.max(0, Math.sin(t * 6.5)) * 0.32;
      addAll(buf, w, {
        rightShoulder: [0.04, -0.05, -0.12],
        spine: [0.04, -0.07, 0.06],
        chest: [0.03, -0.05, 0.04],
        head: [-0.07, -0.04, 0.07],
        neck: [-0.03, -0.02, 0.03],
      });
      // Right arm pumps clenched fist
      arm(buf, 'right', w, {
        raise: 0.88,
        fwd: 0.68,
        twist: -0.78,
        flex: 1.95 - pump,
        hand: [0.18, -0.22, 0.10],
        fingers: FIST_CASCADE,
      });
      arm(buf, 'left', w * 0.6, {
        raise: 0.32,
        fwd: 0.45,
        twist: 0.20,
        flex: 1.15,
        hand: [0.05, 0.12, 0],
        fingers: REST_CASCADE,
      });
    },
  },

  tilt: {
    duration: 2.4,
    expr: { surprised: 0.3, relaxed: 0.35 },
    apply(buf, { w }) {
      addAll(buf, w, {
        head: [0.06, 0.08, 0.22],
        neck: [0.03, 0.04, 0.10],
        spine: [0.02, 0, -0.07],
        chest: [0.01, 0, -0.04],
        leftShoulder: [0, 0, -0.04],
        rightShoulder: [0, 0, -0.08],
      });
      arm(buf, 'left', w * 0.3, {
        raise: 0.30,
        fwd: 0.12,
        flex: 0.22,
        hand: [0.05, 0, 0],
        fingers: REST_CASCADE,
      });
      arm(buf, 'right', w * 0.3, {
        raise: 0.30,
        fwd: 0.12,
        flex: 0.22,
        hand: [0.05, 0, 0],
        fingers: REST_CASCADE,
      });
    },
  },

  shrug: {
    duration: 2.5,
    expr: { relaxed: 0.5, surprised: 0.25 },
    apply(buf, { w }) {
      addAll(buf, w, {
        leftShoulder: [0.02, 0.05, 0.18],
        rightShoulder: [0.02, -0.05, -0.18],
        chest: [-0.04, 0, 0],
        spine: [0.02, 0, 0],
        head: [0.09, 0.05, 0.09],
        neck: [0.04, 0.02, 0.04],
      });
      // Both arms: elbows tucked in, forearms outward, hands supinated open
      arm(buf, 'left', w, {
        raise: 0.45,
        fwd: 0.48,
        twist: 0.68,
        flex: 1.38,
        hand: [0.12, 0.28, 0.18],
        fingers: OPEN_CASCADE,
      });
      arm(buf, 'right', w, {
        raise: 0.45,
        fwd: 0.48,
        twist: -0.68,
        flex: 1.38,
        hand: [0.12, -0.28, -0.18],
        fingers: OPEN_CASCADE,
      });
    },
  },

  surprise: {
    duration: 2.2,
    expr: { surprised: 1.0 },
    apply(buf, { w }) {
      addAll(buf, w, {
        spine: [-0.14, 0, 0],
        chest: [-0.09, 0, 0],
        head: [-0.15, 0, 0],
        neck: [-0.07, 0, 0],
        leftShoulder: [0.06, 0, 0.14],
        rightShoulder: [0.06, 0, -0.14],
      });
      arm(buf, 'left', w, {
        raise: 0.98,
        fwd: 0.42,
        twist: 0.28,
        flex: 0.88,
        hand: [-0.12, 0.15, 0.08],
        fingers: WIDE_CASCADE,
      });
      arm(buf, 'right', w, {
        raise: 0.98,
        fwd: 0.42,
        twist: -0.28,
        flex: 0.88,
        hand: [-0.12, -0.15, -0.08],
        fingers: WIDE_CASCADE,
      });
    },
  },

  relax: {
    duration: 3.0,
    expr: { relaxed: 0.95 },
    apply(buf, { w, t }) {
      const sigh = Math.sin(clamp(t / 3, 0, 1) * Math.PI);
      addAll(buf, w, {
        spine: [0.06 * sigh, 0, 0],
        head: [0.10 * sigh, 0, 0.08 * sigh],
        chest: [-0.04 * sigh, 0, 0],
        leftShoulder: [0, 0, -0.12 * sigh],
        rightShoulder: [0, 0, 0.12 * sigh],
      });
    },
  },

  // Generative Spatial Anchors (Directive #57)
  chin_touch: {
    duration: 3.2,
    expr: { relaxed: 0.45 },
    apply(buf, { w, t }) {
      const breath = Math.sin(t * 0.7) * 0.02;
      addAll(buf, w, {
        rightShoulder: [0.03, 0.04, -0.08],
        head: [-0.08, 0.12 + breath, -0.10],
        neck: [-0.03, 0.05, -0.04],
        spine: [0.03, 0.03, -0.04],
      });
      arm(buf, 'right', w, {
        raise: 0.50,
        fwd: 0.90,
        twist: -0.42,
        flex: 2.40,
        hand: [0.25, -0.40, 0.15],
        fingers: CUPPED_CASCADE,
      });
      arm(buf, 'left', w * 0.7, {
        raise: 0.30,
        fwd: 0.65,
        twist: 0.30,
        flex: 1.45,
        hand: [0.10, 0.12, 0],
        fingers: REST_CASCADE,
      });
    },
  },

  mouth_cover: {
    duration: 2.6,
    expr: { happy: 0.6, relaxed: 0.3 },
    apply(buf, { w, t }) {
      const giggle = Math.sin(t * 12) * 0.02;
      addAll(buf, w, {
        rightShoulder: [0.04, 0.03, -0.09],
        head: [0.08 + giggle, 0.05, 0.08],
        spine: [0.04, 0, 0.03],
      });
      arm(buf, 'right', w, {
        raise: 0.55,
        fwd: 0.85,
        twist: -0.65,
        flex: 2.45,
        hand: [0.30, -0.45, 0.20],
        fingers: CUPPED_CASCADE,
      });
      arm(buf, 'left', w * 0.4, {
        raise: 0.28,
        fwd: 0.30,
        flex: 0.50,
        hand: [0.05, 0.10, 0],
        fingers: REST_CASCADE,
      });
    },
  },

  hand_on_heart: {
    duration: 2.8,
    expr: { happy: 0.45, relaxed: 0.55 },
    apply(buf, { w }) {
      addAll(buf, w, {
        rightShoulder: [0.02, 0.04, -0.06],
        head: [0.06, 0, 0],
        neck: [0.03, 0, 0],
        spine: [0.04, 0, 0],
      });
      arm(buf, 'right', w, {
        raise: 0.52,
        fwd: 0.80,
        twist: -0.50,
        flex: 2.10,
        hand: [0.20, -0.30, 0.15],
        fingers: CUPPED_CASCADE,
      });
      arm(buf, 'left', w * 0.4, {
        raise: 0.26,
        fwd: 0.16,
        flex: 0.30,
        hand: [0.05, 0, 0],
        fingers: REST_CASCADE,
      });
    },
  },

  temple_rub: {
    duration: 3.0,
    expr: { relaxed: 0.4 },
    apply(buf, { w, t }) {
      const circle = Math.sin(t * 5) * 0.03;
      addAll(buf, w, {
        rightShoulder: [0.04, 0.03, -0.09],
        head: [-0.07, 0.10, -0.08],
        neck: [-0.02, 0.04, -0.03],
        spine: [0.03, 0.02, -0.03],
      });
      arm(buf, 'right', w, {
        raise: 0.65,
        fwd: 0.75,
        twist: -0.55,
        flex: 2.50 + circle,
        hand: [0.35, -0.40, 0.10],
        fingers: POINT_CASCADE,
      });
    },
  },

  embrace: {
    duration: 3.0,
    expr: { happy: 0.8, relaxed: 0.4 },
    apply(buf, { w }) {
      addAll(buf, w, {
        leftShoulder: [0.02, 0.05, 0.08],
        rightShoulder: [0.02, -0.05, -0.08],
        chest: [0.06, 0, 0],
        spine: [0.03, 0, 0],
        head: [-0.04, 0, 0],
      });
      arm(buf, 'left', w, {
        raise: 0.65,
        fwd: 0.75,
        twist: 0.35,
        flex: 0.85,
        hand: [0, 0.20, 0.15],
        fingers: OPEN_CASCADE,
      });
      arm(buf, 'right', w, {
        raise: 0.65,
        fwd: 0.75,
        twist: -0.35,
        flex: 0.85,
        hand: [0, -0.20, -0.15],
        fingers: OPEN_CASCADE,
      });
    },
  },

  rub_hands: {
    duration: 2.6,
    expr: { happy: 0.55 },
    apply(buf, { w, t }) {
      const rub = Math.sin(t * 12) * 0.05;
      addAll(buf, w, {
        chest: [0.04, 0, 0],
        head: [0.05, 0, 0],
      });
      arm(buf, 'left', w, {
        raise: 0.48,
        fwd: 0.95,
        twist: 0.40,
        flex: 1.85 + rub,
        hand: [0.10, 0.30, 0],
        fingers: CUPPED_CASCADE,
      });
      arm(buf, 'right', w, {
        raise: 0.48,
        fwd: 0.95,
        twist: -0.40,
        flex: 1.85 - rub,
        hand: [0.10, -0.30, 0],
        fingers: CUPPED_CASCADE,
      });
    },
  },

  lean_forward: {
    duration: 2.8,
    expr: { happy: 0.3, relaxed: 0.3 },
    apply(buf, { w }) {
      addAll(buf, w, {
        spine: [0.15, 0, 0],
        chest: [0.10, 0, 0],
        head: [-0.08, 0, 0],
        neck: [-0.03, 0, 0],
      });
    },
  },

  lean_back: {
    duration: 2.8,
    expr: { relaxed: 0.65 },
    apply(buf, { w }) {
      addAll(buf, w, {
        spine: [-0.12, 0, 0],
        chest: [-0.08, 0, 0],
        head: [0.06, 0, 0],
        neck: [0.03, 0, 0],
      });
    },
  },
};

const ALIASES: Record<string, string> = {
  cheer: 'praise',
  clapping: 'clap',
  greet: 'wave',
  hello: 'wave',
  ok: 'nod',
  yes: 'nod',
  no: 'shake',
  sigh: 'relax',
  confused: 'think',
  salute: 'wave',
  laugh: 'praise',
  idle: 'relax',
  dance: 'praise',
  wink: 'wave',

  headtilt: 'tilt',
  tilthead: 'tilt',
  shouldershrug: 'shrug',
  shrugging: 'shrug',

  // Vietnamese natural language mappings
  chao: 'wave',
  vaytay: 'wave',
  vaytaychao: 'wave',
  xinchao: 'wave',
  giaithich: 'explain',
  trinhbay: 'explain',
  chitay: 'point',
  chihuong: 'point',
  suynghi: 'think',
  tramngam: 'think',
  ngamnghi: 'think',
  yituong: 'idea',
  sangtao: 'idea',
  gatdau: 'nod',
  dongy: 'nod',
  lacdau: 'shake',
  tuchoi: 'shake',
  khenngoi: 'praise',
  hoanho: 'praise',
  votay: 'clap',
  cuidau: 'bow',
  cuichao: 'bow',
  nguongngung: 'shy',
  eap: 'shy',
  xauho: 'shy',
  dongvien: 'encourage',
  cotvu: 'encourage',
  quyetam: 'encourage',
  nghiengdau: 'tilt',
  nhunvai: 'shrug',
  ngacnhien: 'surprise',
  giatminh: 'surprise',
  thugian: 'relax',
  thathalong: 'relax',

  // Generative anatomical anchors (Vietnamese & English)
  chongcam: 'chin_touch',
  aptaylenma: 'chin_touch',
  chintouch: 'chin_touch',
  touchchin: 'chin_touch',
  chemiengcuoi: 'mouth_cover',
  chemieng: 'mouth_cover',
  bitmieng: 'mouth_cover',
  mouthcover: 'mouth_cover',
  covermouth: 'mouth_cover',
  dattaylentim: 'hand_on_heart',
  chaptaytruocnguc: 'hand_on_heart',
  handonheart: 'hand_on_heart',
  daythaiduong: 'temple_rub',
  chitran: 'temple_rub',
  templerub: 'temple_rub',
  rubtemple: 'temple_rub',
  duahaitayratruoc: 'embrace',
  om: 'embrace',
  hug: 'embrace',
  xoahaibantay: 'rub_hands',
  rubhands: 'rub_hands',
  cuilaigan: 'lean_forward',
  leanforward: 'lean_forward',
  nganguoirasau: 'lean_back',
  leanback: 'lean_back',
};

const HEAD_ONLY = new Set(['nod', 'shake', 'relax', 'tilt', 'lean_forward', 'lean_back']);
const MAX_QUEUE = 4;
const OVERLAP = 0.4;

interface ActiveGesture {
  key: string;
  def: GestureDef;
  duration: number;
  time: number;
}

interface GestureRequest {
  key: string;
  duration?: number;
}

class Lane {
  public active: ActiveGesture[] = [];
  public queue: GestureRequest[] = [];
}

export class GestureController {
  public validGestures = Object.keys(GESTURES);
  private lanes = { head: new Lane(), body: new Lane() };
  private attack = 0.28;
  private release = OVERLAP;

  // Zero-allocation pre-allocated state pool
  private _statePool: Array<{ key: string; def: GestureDef; w: number; t: number }> = Array.from(
    { length: 16 },
    () => ({ key: '', def: GESTURES.wave, w: 0, t: 0 })
  );
  private _activeStates: Array<{ key: string; def: GestureDef; w: number; t: number }> = [];
  private _hint: Record<string, number> = {};

  // Co-speech prosody gesticulation state
  public isSpeaking = false;
  public mouthEnergy = 0;
  private speechTime = 0;

  public resolve(name: string): string | null {
    if (!name) return null;
    const key = String(name).toLowerCase().replace(/[-_\s*()]/g, '');
    return GESTURES[key] ? key : ALIASES[key] || null;
  }

  private laneFor(key: string): Lane {
    return HEAD_ONLY.has(key) ? this.lanes.head : this.lanes.body;
  }

  public playGesture(name: string, duration?: number): boolean {
    const key = this.resolve(name);
    if (!key) return false;
    const lane = this.laneFor(key);
    const request: GestureRequest = { key, duration: duration ?? GESTURES[key].duration };

    const last = lane.queue[lane.queue.length - 1] || lane.active[lane.active.length - 1];
    if (last && last.key === key) {
      if ((last as ActiveGesture).time !== undefined) {
        (last as ActiveGesture).duration = Math.min(
          (last as ActiveGesture).duration + 0.8,
          (last as ActiveGesture).duration + (request.duration || 2.0) * 0.3
        );
      }
      return true;
    }

    lane.queue.push(request);
    while (lane.queue.length > MAX_QUEUE) lane.queue.shift();
    if (!lane.active.length) this.startNext(lane);
    return true;
  }

  private startNext(lane: Lane): void {
    const next = lane.queue.shift();
    if (!next) return;
    lane.active.push({
      key: next.key,
      def: GESTURES[next.key],
      duration: next.duration || GESTURES[next.key].duration,
      time: 0,
    });
  }

  public isGesturing(): boolean {
    return Object.values(this.lanes).some((l) => l.active.length || l.queue.length);
  }

  public setSpeaking(speaking: boolean, mouthEnergy = 0): void {
    this.isSpeaking = speaking;
    this.mouthEnergy = clamp(mouthEnergy, 0, 1);
    if (!speaking) {
      this.speechTime = 0;
    }
  }

  public stop(): void {
    for (const lane of Object.values(this.lanes)) {
      lane.queue = [];
      for (const g of lane.active) {
        g.duration = Math.min(g.duration, g.time + this.release);
      }
    }
  }

  public clear(): void {
    for (const lane of Object.values(this.lanes)) {
      lane.active = [];
      lane.queue = [];
    }
  }

  public update(
    dt: number,
    mouthEnergy = 0,
    isSpeaking?: boolean
  ): Array<{ key: string; def: GestureDef; w: number; t: number }> {
    if (isSpeaking !== undefined) this.isSpeaking = isSpeaking;
    if (mouthEnergy > 0 || isSpeaking) this.mouthEnergy = clamp(mouthEnergy, 0, 1);

    if (this.isSpeaking) {
      this.speechTime += dt;
    }

    this._activeStates.length = 0;
    let poolIndex = 0;

    const laneValues = [this.lanes.head, this.lanes.body];
    for (let l = 0; l < laneValues.length; l++) {
      const lane = laneValues[l];
      for (let i = 0; i < lane.active.length; i++) {
        const g = lane.active[i];
        g.time += dt;

        // 5th-order minimum-jerk polynomial for C^2 continuous transitions
        const wAttack = quintic(g.time / this.attack);
        const wRelease = quintic((g.duration - g.time) / this.release);
        const w = Math.min(wAttack, wRelease);

        if (poolIndex < this._statePool.length) {
          const st = this._statePool[poolIndex++];
          st.key = g.key;
          st.def = g.def;
          st.w = w;
          st.t = g.time;
          this._activeStates.push(st);
        }
      }

      // In-place two-pointer compaction to eliminate array allocations
      let writeIdx = 0;
      for (let readIdx = 0; readIdx < lane.active.length; readIdx++) {
        const g = lane.active[readIdx];
        if (g.time < g.duration) {
          lane.active[writeIdx++] = g;
        }
      }
      lane.active.length = writeIdx;

      const newest = lane.active[lane.active.length - 1];
      if (lane.queue.length && (!newest || newest.time >= newest.duration - this.release)) {
        this.startNext(lane);
      }
    }

    return this._activeStates;
  }

  public apply(
    buf: PoseBuffer,
    states: Array<{ key: string; def: GestureDef; w: number; t: number }>,
    options?: { mouthEnergy?: number; isSpeaking?: boolean; time?: number }
  ): Record<string, number> {
    // Clear pre-allocated hint record without heap allocation
    for (const k in this._hint) {
      delete this._hint[k];
    }

    let maxGestureWeight = 0;
    for (let i = 0; i < states.length; i++) {
      const state = states[i];
      state.def.apply(buf, state);
      if (state.w > maxGestureWeight) maxGestureWeight = state.w;
      const expr = state.def.expr;
      if (expr) {
        for (const k in expr) {
          this._hint[k] = Math.max(this._hint[k] || 0, expr[k] * state.w);
        }
      }
    }

    // Co-Speech Prosody Gesticulation Engine
    const speaking = options?.isSpeaking ?? this.isSpeaking;
    const energy = options?.mouthEnergy ?? this.mouthEnergy;
    const time = options?.time ?? this.speechTime;

    if (speaking && (energy > 0.05 || time > 0)) {
      this.applyCoSpeech(buf, time, energy, maxGestureWeight);
    }

    return this._hint;
  }

  /**
   * Generates organic co-speech beat nods, rhythmic hand pulses,
   * and shoulder emphasis modulated by vocal energy with non-repetitive micro-variations.
   */
  public applyCoSpeech(
    buf: PoseBuffer,
    time: number,
    mouthEnergy: number,
    activeGestureWeight = 0
  ): void {
    // Co-speech weight gracefully yields when explicit gestures are active
    const coSpeechWeight = (1 - activeGestureWeight * 0.75) * clamp(mouthEnergy * 1.5, 0.1, 1.0);
    if (coSpeechWeight <= 0.005) return;

    // 1. Organic Co-Speech Beat Nods (dual incommensurate frequencies + vocal stress peaks)
    const beatNod =
      mouthEnergy * (0.045 * Math.sin(time * 11.5) + 0.025 * Math.sin(time * 7.3)) +
      0.015 * noise(time * 1.8, 41);
    const beatYaw = 0.025 * mouthEnergy * Math.sin(time * 3.1) + 0.012 * noise(time * 1.2, 43);
    const beatRoll = 0.018 * mouthEnergy * Math.sin(time * 2.3 + 0.5);

    buf.add('head', beatNod * coSpeechWeight, beatYaw * coSpeechWeight, beatRoll * coSpeechWeight);
    buf.add('neck', beatNod * 0.5 * coSpeechWeight, beatYaw * 0.35 * coSpeechWeight, 0);

    // 2. Shoulder Emphasis (acoustic peaks elevate clavicles)
    const shoulderBeat = mouthEnergy * 0.035 + 0.012 * noise(time * 2.2, 47);
    buf.add('leftShoulder', 0, 0, shoulderBeat * coSpeechWeight);
    buf.add('rightShoulder', 0, 0, -shoulderBeat * coSpeechWeight);

    // 3. Rhythmic Hand Pulses & Conversational Batonic Gestures
    if (activeGestureWeight < 0.3) {
      const armPulseR = mouthEnergy * (0.12 * Math.sin(time * 6.2) + 0.06 * noise(time * 1.5, 51));
      const armPulseL = mouthEnergy * (0.10 * Math.sin(time * 5.8 + 1.2) + 0.05 * noise(time * 1.4, 53));

      buf.add('rightLowerArm', 0, -armPulseR * coSpeechWeight * 0.45, 0);
      buf.add('leftLowerArm', 0, armPulseL * coSpeechWeight * 0.45, 0);
      buf.add('rightHand', 0.05 * mouthEnergy * coSpeechWeight, -armPulseR * 0.25 * coSpeechWeight, 0);
      buf.add('leftHand', 0.05 * mouthEnergy * coSpeechWeight, armPulseL * 0.25 * coSpeechWeight, 0);

      // Micro-curl conversational finger pulsing
      const fingerPulse = mouthEnergy * 0.06 * Math.sin(time * 6.0);
      buf.setFingers('right', {
        ...REST_CASCADE,
        index: clamp(0.20 + fingerPulse, 0.10, 0.40),
        middle: clamp(0.35 + fingerPulse, 0.20, 0.55),
      });
      buf.setFingers('left', {
        ...REST_CASCADE,
        index: clamp(0.20 + fingerPulse, 0.10, 0.40),
        middle: clamp(0.35 + fingerPulse, 0.20, 0.55),
      });
    }
  }
}
