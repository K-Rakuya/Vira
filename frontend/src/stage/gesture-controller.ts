import * as THREE from 'three';
import { clamp, smoothstep, eq, upperArmQuat, lowerArmQuat, type Side, type PoseBuffer, type FingerCurls } from './pose-utils.js';

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

const arm = (buf: PoseBuffer, side: Side, w: number, spec: ArmSpec) => {
  const { raise = 0.3, fwd = 0.1, twist = 0, flex = 0.2, ftwist = 0, hand, fingers } = spec;
  const prefix = side;
  buf.blend(`${prefix}UpperArm`, upperArmQuat(side, raise, fwd, twist, _q), w);
  buf.blend(`${prefix}LowerArm`, lowerArmQuat(side, flex, ftwist, _q), w);
  if (hand) buf.blend(`${prefix}Hand`, eq(hand[0], hand[1], hand[2], _q), w);
  if (fingers) {
    const merged: FingerCurls = {};
    for (const [k, v] of Object.entries(fingers)) {
      if (typeof v === 'number') {
        merged[k as keyof FingerCurls] = v * w + (1 - w) * (fingers.rest ?? 0.25);
      }
    }
    buf.setFingers(side, merged);
  }
};

const addAll = (buf: PoseBuffer, w: number, map: Record<string, [number, number, number]>) => {
  for (const [bone, [x, y, z]] of Object.entries(map)) {
    buf.add(bone, x * w, y * w, z * w);
  }
};

const OPEN: FingerCurls = { all: 0.05, thumb: 0.05 };
const FIST: FingerCurls = { all: 1.0, thumb: 0.9 };

export interface GestureDef {
  duration: number;
  expr?: Record<string, number>;
  apply: (buf: PoseBuffer, ctx: { w: number; t: number }) => void;
}

export const GESTURES: Record<string, GestureDef> = {
  wave: {
    duration: 2.6,
    expr: { happy: 0.35 },
    apply(buf, { w, t }) {
      const flap = Math.sin(t * 11) * 0.28;
      arm(buf, 'right', w, { raise: 1.45, fwd: 0.2, twist: -1.35, flex: 1.05 + flap * 0.4, hand: [0, 0, flap * 0.5], fingers: { ...OPEN } });
      addAll(buf, w, { head: [0, 0, -0.08], spine: [0, 0, -0.03] });
    },
  },
  explain: {
    duration: 2.8,
    expr: { relaxed: 0.35 },
    apply(buf, { w, t }) {
      const pulse = Math.sin(t * 3.2) * 0.12;
      arm(buf, 'left', w, { raise: 0.55, fwd: 0.65, twist: 0.5, flex: 1.15 + pulse, hand: [0, 0, 0.1], fingers: { ...OPEN, rest: 0.25 } });
      arm(buf, 'right', w, { raise: 0.55, fwd: 0.65, twist: -0.5, flex: 1.15 - pulse, hand: [0, 0, -0.1], fingers: { ...OPEN, rest: 0.25 } });
      addAll(buf, w, { chest: [0.05, 0, 0], head: [0.02, Math.sin(t * 1.5) * 0.08, 0] });
    },
  },
  point: {
    duration: 2.2,
    expr: { happy: 0.2 },
    apply(buf, { w }) {
      arm(buf, 'right', w, { raise: 0.85, fwd: 1.5, twist: 0, flex: 0.35, fingers: { index: 0, thumb: 0.4, middle: 1, ring: 1, little: 1, rest: 0.25 } });
      addAll(buf, w, { head: [-0.03, -0.1, 0], chest: [0.02, -0.05, 0] });
    },
  },
  think: {
    duration: 3.0,
    expr: { relaxed: 0.2 },
    apply(buf, { w, t }) {
      arm(buf, 'right', w, { raise: 0.45, fwd: 0.9, twist: -0.4, flex: 2.35, fingers: { all: 0.6, index: 0.2, rest: 0.25 } });
      arm(buf, 'left', w, { raise: 0.3, fwd: 0.7, twist: 0.3, flex: 1.5, fingers: { all: 0.45, rest: 0.25 } });
      addAll(buf, w, { head: [-0.1, 0.12 + Math.sin(t * 0.9) * 0.03, -0.1], neck: [-0.03, 0.05, 0] });
    },
  },
  idea: {
    duration: 2.4,
    expr: { happy: 0.5 },
    apply(buf, { w, t }) {
      const bob = Math.sin(t * 7) * 0.05;
      arm(buf, 'right', w, { raise: 1.1, fwd: 0.5, twist: -0.6, flex: 1.5 + bob, fingers: { index: 0, thumb: 0.5, middle: 1, ring: 1, little: 1, rest: 0.25 } });
      addAll(buf, w, { head: [-0.08, 0, 0], spine: [-0.05, 0, 0] });
    },
  },
  nod: {
    duration: 1.8,
    expr: { happy: 0.2 },
    apply(buf, { w, t }) {
      const n = Math.sin(t * 8) * 0.17;
      addAll(buf, w, { head: [n, 0, 0], neck: [n * 0.5, 0, 0] });
    },
  },
  shake: {
    duration: 1.8,
    expr: { sad: 0.3 },
    apply(buf, { w, t }) {
      const s = Math.sin(t * 9) * 0.26;
      addAll(buf, w, { head: [0, s, 0], neck: [0, s * 0.5, 0] });
    },
  },
  praise: {
    duration: 2.6,
    expr: { happy: 1 },
    apply(buf, { w, t }) {
      const bounce = Math.abs(Math.sin(t * 6));
      arm(buf, 'left', w, { raise: 2.5, fwd: 0.15, twist: 0.3, flex: 0.35, fingers: { ...OPEN } });
      arm(buf, 'right', w, { raise: 2.5, fwd: 0.15, twist: -0.3, flex: 0.35, fingers: { ...OPEN } });
      buf.offsetPos('hips', 0, bounce * 0.025 * w, 0);
      addAll(buf, w, { head: [-0.1, 0, 0], chest: [-0.06, 0, 0] });
    },
  },
  clap: {
    duration: 2.6,
    expr: { happy: 0.5 },
    apply(buf, { w, t }) {
      const c = (Math.sin(t * 13) * 0.5 + 0.5) * 0.35;
      arm(buf, 'left', w, { raise: 0.5, fwd: 1.2, twist: 0.4, flex: 1.75 - c * 0.3, hand: [0, 0.2 + c * 0.2, 0], fingers: { ...OPEN, rest: 0.25 } });
      arm(buf, 'right', w, { raise: 0.5, fwd: 1.2, twist: -0.4, flex: 1.75 - c * 0.3, hand: [0, -0.2 - c * 0.2, 0], fingers: { ...OPEN, rest: 0.25 } });
      addAll(buf, w, { head: [0.05, 0, 0] });
    },
  },
  bow: {
    duration: 2.8,
    expr: { relaxed: 0.7 },
    apply(buf, { w }) {
      arm(buf, 'left', w, { raise: 0.18, fwd: 0.9, twist: 0.2, flex: 0.9, fingers: { all: 0.25 } });
      arm(buf, 'right', w, { raise: 0.18, fwd: 0.9, twist: -0.2, flex: 0.9, fingers: { all: 0.25 } });
      addAll(buf, w, { spine: [0.32, 0, 0], chest: [0.2, 0, 0], neck: [0.1, 0, 0], head: [0.1, 0, 0] });
    },
  },
  shy: {
    duration: 2.8,
    expr: { happy: 0.25, relaxed: 0.3 },
    apply(buf, { w, t }) {
      const sway = Math.sin(t * 2) * 0.03;
      arm(buf, 'left', w, { raise: 0.3, fwd: 0.85, twist: 0.2, flex: 1.7 + sway, fingers: { all: 0.35 } });
      arm(buf, 'right', w, { raise: 0.3, fwd: 0.85, twist: -0.2, flex: 1.7 - sway, fingers: { all: 0.35 } });
      addAll(buf, w, { head: [0.12, 0.12, 0.14], spine: [0.05, 0, 0.03] });
    },
  },
  encourage: {
    duration: 2.4,
    expr: { happy: 0.5 },
    apply(buf, { w, t }) {
      const pump = Math.max(0, Math.sin(t * 6)) * 0.3;
      arm(buf, 'right', w, { raise: 0.95, fwd: 0.6, twist: -0.8, flex: 1.9 - pump, fingers: { ...FIST, rest: 0.25 } });
      addAll(buf, w, { head: [-0.06, 0, 0.05], chest: [0, -0.05, 0] });
    },
  },
  surprise: {
    duration: 2.2,
    expr: { surprised: 1 },
    apply(buf, { w }) {
      arm(buf, 'left', w, { raise: 0.95, fwd: 0.3, twist: 0.3, flex: 0.7, fingers: { ...OPEN } });
      arm(buf, 'right', w, { raise: 0.95, fwd: 0.3, twist: -0.3, flex: 0.7, fingers: { ...OPEN } });
      addAll(buf, w, { spine: [-0.1, 0, 0], chest: [-0.06, 0, 0], head: [-0.12, 0, 0] });
    },
  },
  relax: {
    duration: 3.0,
    expr: { relaxed: 0.9 },
    apply(buf, { w, t }) {
      const sigh = Math.sin(clamp(t / 3, 0, 1) * Math.PI);
      addAll(buf, w, { spine: [0.05 * sigh, 0, 0], head: [0.09, 0, 0.07], chest: [-0.03 * sigh, 0, 0] });
      buf.add('leftShoulder', 0, 0, -0.1 * w * sigh);
      buf.add('rightShoulder', 0, 0, 0.1 * w * sigh);
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
};

const HEAD_ONLY = new Set(['nod', 'shake', 'relax']);
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

  public resolve(name: string): string | null {
    if (!name) return null;
    const key = String(name).toLowerCase().replace(/[-_\s]/g, '');
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
        (last as ActiveGesture).duration = Math.min((last as ActiveGesture).duration + 0.8, (last as ActiveGesture).duration + (request.duration || 2.0) * 0.3);
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
    lane.active.push({ key: next.key, def: GESTURES[next.key], duration: next.duration || GESTURES[next.key].duration, time: 0 });
  }

  public isGesturing(): boolean {
    return Object.values(this.lanes).some((l) => l.active.length || l.queue.length);
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

  public update(dt: number): Array<{ key: string; def: GestureDef; w: number; t: number }> {
    const states: Array<{ key: string; def: GestureDef; w: number; t: number }> = [];
    for (const lane of Object.values(this.lanes)) {
      for (const g of lane.active) {
        g.time += dt;
        const w = Math.min(smoothstep(g.time / this.attack), smoothstep((g.duration - g.time) / this.release));
        states.push({ key: g.key, def: g.def, w, t: g.time });
      }
      lane.active = lane.active.filter((g) => g.time < g.duration);

      const newest = lane.active[lane.active.length - 1];
      if (lane.queue.length && (!newest || newest.time >= newest.duration - this.release)) {
        this.startNext(lane);
      }
    }
    return states;
  }

  public apply(buf: PoseBuffer, states: Array<{ key: string; def: GestureDef; w: number; t: number }>): Record<string, number> {
    const hint: Record<string, number> = {};
    for (const state of states || []) {
      state.def.apply(buf, state);
      for (const [k, v] of Object.entries(state.def.expr || {})) {
        hint[k] = Math.max(hint[k] || 0, v * state.w);
      }
    }
    return hint;
  }
}
