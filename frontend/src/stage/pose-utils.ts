import * as THREE from 'three';
import type { VRM, VRMHumanBoneName } from '@pixiv/three-vrm';

const _e = new THREE.Euler();
const _qa = new THREE.Quaternion();
const _qb = new THREE.Quaternion();
const _va = new THREE.Vector3();
const _vb = new THREE.Vector3();
const IDENTITY = new THREE.Quaternion();

export const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));
export const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;
export const smoothstep = (t: number): number => {
  const x = clamp(t, 0, 1);
  return x * x * (3 - 2 * x);
};

export function eq(x: number, y: number, z: number, out = new THREE.Quaternion()): THREE.Quaternion {
  _e.set(x, y, z, 'XYZ');
  return out.setFromEuler(_e);
}

export function noise(t: number, seed = 0): number {
  return (
    Math.sin(t * 1.0 + seed * 12.9898) * 0.5 +
    Math.sin(t * 2.3 + seed * 78.233) * 0.3 +
    Math.sin(t * 4.1 + seed * 37.719) * 0.2
  );
}

export type Side = 'left' | 'right';
export const sideSign = (side: Side): number => (side === 'left' ? 1 : -1);

export function upperArmQuat(
  side: Side,
  raise: number,
  forward = 0,
  twist = 0,
  out = new THREE.Quaternion()
): THREE.Quaternion {
  const s = sideSign(side);
  _va.set(s * Math.sin(raise), -Math.cos(raise), forward).normalize();
  _vb.set(s, 0, 0);
  out.setFromUnitVectors(_vb, _va);
  if (twist) out.multiply(_qa.setFromAxisAngle(_vb.set(1, 0, 0), -twist * s));
  return out;
}

export function lowerArmQuat(
  side: Side,
  flex: number,
  twist = 0,
  out = new THREE.Quaternion()
): THREE.Quaternion {
  const s = sideSign(side);
  _e.set(0, -s * flex, 0, 'XYZ');
  out.setFromEuler(_e);
  if (twist) out.multiply(_qa.setFromAxisAngle(_vb.set(1, 0, 0), -twist * s));
  return out;
}

const FINGERS = ['Index', 'Middle', 'Ring', 'Little'] as const;

export interface FingerCurls {
  all?: number;
  thumb?: number;
  index?: number;
  middle?: number;
  ring?: number;
  little?: number;
  rest?: number;
}

export class PoseBuffer {
  public rot = new Map<string, THREE.Quaternion>();
  public pos = new Map<string, [number, number, number]>();
  public smoothed = new Map<string, THREE.Quaternion>();
  public smoothedPos = new Map<string, [number, number, number]>();

  public reset(): void {
    this.rot.clear();
    this.pos.clear();
  }

  public clearSmoothing(): void {
    this.smoothed.clear();
    this.smoothedPos.clear();
  }

  public get(bone: string): THREE.Quaternion {
    let q = this.rot.get(bone);
    if (!q) {
      q = new THREE.Quaternion();
      this.rot.set(bone, q);
    }
    return q;
  }

  public set(bone: string, q: THREE.Quaternion): void {
    this.get(bone).copy(q);
  }

  public add(bone: string, x: number, y: number, z: number): void {
    this.get(bone).multiply(eq(x, y, z, _qb));
  }

  public blend(bone: string, q: THREE.Quaternion, w: number): void {
    if (w <= 0) return;
    this.get(bone).slerp(q, clamp(w, 0, 1));
  }

  public offsetPos(bone: string, x: number, y: number, z: number): void {
    const p = this.pos.get(bone) || [0, 0, 0];
    p[0] += x;
    p[1] += y;
    p[2] += z;
    this.pos.set(bone, p);
  }

  public setFingers(side: Side, curls: FingerCurls): void {
    const s = sideSign(side);
    const prefix = side;
    FINGERS.forEach((name) => {
      const key = name.toLowerCase() as keyof FingerCurls;
      const c = curls[key] ?? curls.all ?? 0;
      const joints = ['Proximal', 'Intermediate', 'Distal'];
      const gains = [1.1, 1.3, 0.8];
      joints.forEach((j, i) => {
        this.get(`${prefix}${name}${j}`).copy(eq(0, 0, -s * c * gains[i], _qb));
      });
    });
    const t = curls.thumb ?? (curls.all ?? 0) * 0.5;
    this.get(`${prefix}ThumbProximal`).copy(eq(0, s * t * 0.35, -s * t * 0.2, _qb));
    this.get(`${prefix}ThumbDistal`).copy(eq(0, 0, -s * t * 0.5, _qb));
  }

  public apply(vrm: VRM, dt: number, smoothing = 14): void {
    const humanoid = vrm?.humanoid;
    if (!humanoid) return;

    const flip = vrm.meta?.metaVersion === '0';
    const k = 1 - Math.exp(-smoothing * Math.max(dt, 0));
    const pose: Record<string, { rotation?: [number, number, number, number]; position?: [number, number, number] }> = {};

    for (const [bone, target] of this.rot) {
      if (!this.smoothed.has(bone)) this.smoothed.set(bone, target.clone());
    }
    for (const [bone, current] of this.smoothed) {
      const target = this.rot.get(bone) || IDENTITY;
      current.slerp(target, k);
      const entry: { rotation?: [number, number, number, number]; position?: [number, number, number] } = {};
      entry.rotation = flip
        ? [-current.x, current.y, -current.z, current.w]
        : [current.x, current.y, current.z, current.w];
      pose[bone] = entry;
    }

    for (const [bone, target] of this.pos) {
      if (!this.smoothedPos.has(bone)) this.smoothedPos.set(bone, [0, 0, 0]);
    }
    for (const [bone, current] of this.smoothedPos) {
      const target = this.pos.get(bone) || [0, 0, 0];
      for (let i = 0; i < 3; i++) current[i] += (target[i] - current[i]) * k;
      const entry = pose[bone] || (pose[bone] = {});
      entry.position = flip ? [-current[0], current[1], -current[2]] : [...current];
    }

    humanoid.resetNormalizedPose();
    humanoid.setNormalizedPose(pose as any);
  }
}
