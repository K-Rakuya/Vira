import * as THREE from 'three';
import type { VRM, VRMHumanBoneName } from '@pixiv/three-vrm';

const _e = new THREE.Euler();
const _qa = new THREE.Quaternion();
const _qb = new THREE.Quaternion();
const _va = new THREE.Vector3();
const _vb = new THREE.Vector3();
const _scratchQuat = new THREE.Quaternion();
const _zeroPos: [number, number, number] = [0, 0, 0];
const IDENTITY = new THREE.Quaternion();

const JOINTS = ['Proximal', 'Intermediate', 'Distal'] as const;
const GAINS = [1.1, 1.3, 0.8] as const;

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

/**
 * Maps a unit quaternion q in SO(3) to its Lie algebra vector in so(3) ~= R^3.
 * theta_vec = Log(q)
 */
export function so3Log(q: THREE.Quaternion, out = new THREE.Vector3()): THREE.Vector3 {
  let w = clamp(q.w, -1, 1);
  let x = q.x;
  let y = q.y;
  let z = q.z;
  if (w < 0) {
    w = -w;
    x = -x;
    y = -y;
    z = -z;
  }
  const vLen = Math.hypot(x, y, z);
  if (vLen < 1e-7) {
    return out.set(2 * x, 2 * y, 2 * z);
  }
  const angle = 2 * Math.atan2(vLen, w);
  const factor = angle / vLen;
  return out.set(x * factor, y * factor, z * factor);
}

/**
 * Maps a Lie algebra vector in so(3) ~= R^3 to its unit quaternion in SO(3).
 * q = Exp(v)
 */
export function so3Exp(v: THREE.Vector3, out = new THREE.Quaternion()): THREE.Quaternion {
  const theta = v.length();
  if (theta < 1e-7) {
    return out.set(0, 0, 0, 1);
  }
  const half = theta * 0.5;
  const factor = Math.sin(half) / theta;
  return out.set(v.x * factor, v.y * factor, v.z * factor, Math.cos(half)).normalize();
}

/**
 * 2nd-order Geodesic Spring-Damper on Lie Group SO(3).
 * Operates on the Riemannian manifold SO(3) via Lie algebra logarithmic and exponential maps,
 * eliminating Gimbal Lock, singularities, and coordinate artifacts.
 */
export class GeodesicSpringDamper {
  public current = new THREE.Quaternion();
  public velocity = new THREE.Vector3(); // angular velocity in rad/s (in so(3))
  public omegaN: number; // Natural frequency (rad/s)
  public zeta: number; // Damping ratio (1.0 = critical, 0.82 = slight organic compliance)

  private _rel = new THREE.Quaternion();
  private _err = new THREE.Vector3();
  private _acc = new THREE.Vector3();
  private _deltaRot = new THREE.Vector3();
  private _expQ = new THREE.Quaternion();

  constructor(omegaN = 14, zeta = 1.0) {
    this.omegaN = omegaN;
    this.zeta = zeta;
  }

  public reset(q?: THREE.Quaternion): void {
    if (q) this.current.copy(q);
    else this.current.identity();
    this.velocity.set(0, 0, 0);
  }

  public step(target: THREE.Quaternion, dt: number): THREE.Quaternion {
    if (dt <= 0) return this.current;
    const clampedDt = Math.min(dt, 0.05);

    // Relative rotation error: q_rel = q^-1 * q_target
    this._rel.copy(this.current).invert().multiply(target);

    // Map to Lie algebra so(3) error vector
    so3Log(this._rel, this._err);

    // Angular acceleration: alpha = omega_n^2 * theta_err - 2 * zeta * omega_n * velocity
    const kSpring = this.omegaN * this.omegaN;
    const kDamp = 2 * this.zeta * this.omegaN;

    this._acc.copy(this._err).multiplyScalar(kSpring).addScaledVector(this.velocity, -kDamp);

    // Semi-implicit integration
    this.velocity.addScaledVector(this._acc, clampedDt);

    // Incremental rotation: delta_theta = velocity * dt
    this._deltaRot.copy(this.velocity).multiplyScalar(clampedDt);
    so3Exp(this._deltaRot, this._expQ);

    // Update orientation on SO(3): q_new = q * Exp(delta_theta)
    this.current.multiply(this._expQ).normalize();

    return this.current;
  }
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

  private _rotPool: THREE.Quaternion[] = [];
  private _posPool: [number, number, number][] = [];

  public reset(): void {
    for (const q of this.rot.values()) {
      this._rotPool.push(q);
    }
    this.rot.clear();
    for (const p of this.pos.values()) {
      this._posPool.push(p);
    }
    this.pos.clear();
  }

  public clearSmoothing(): void {
    this.smoothed.clear();
    this.smoothedPos.clear();
  }

  public get(bone: string): THREE.Quaternion {
    let q = this.rot.get(bone);
    if (!q) {
      q = this._rotPool.pop() || new THREE.Quaternion();
      q.identity();
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
    let p = this.pos.get(bone);
    if (!p) {
      p = this._posPool.pop() || [0, 0, 0];
      p[0] = 0;
      p[1] = 0;
      p[2] = 0;
      this.pos.set(bone, p);
    }
    p[0] += x;
    p[1] += y;
    p[2] += z;
  }

  public setFingers(side: Side, curls: FingerCurls): void {
    const s = sideSign(side);
    const prefix = side;
    for (let f = 0; f < FINGERS.length; f++) {
      const name = FINGERS[f];
      const key = name.toLowerCase() as keyof FingerCurls;
      const c = curls[key] ?? curls.all ?? 0;
      for (let j = 0; j < JOINTS.length; j++) {
        this.get(`${prefix}${name}${JOINTS[j]}`).copy(eq(0, 0, -s * c * GAINS[j], _qb));
      }
    }
    const t = curls.thumb ?? (curls.all ?? 0) * 0.5;
    this.get(`${prefix}ThumbProximal`).copy(eq(0, s * t * 0.35, -s * t * 0.2, _qb));
    this.get(`${prefix}ThumbDistal`).copy(eq(0, 0, -s * t * 0.5, _qb));
  }

  public apply(vrm: VRM, dt = 0.016, smoothing = 14): void {
    const humanoid = vrm?.humanoid;
    if (!humanoid) return;

    const flip = vrm.meta?.metaVersion === '0';
    const safeDt = typeof dt === 'number' && !isNaN(dt) ? Math.max(dt, 0) : 0.016;
    const k = 1 - Math.exp(-smoothing * safeDt);

    for (const [bone, target] of this.rot) {
      if (!this.smoothed.has(bone)) {
        const sq = new THREE.Quaternion().copy(target);
        this.smoothed.set(bone, sq);
      }
    }
    for (const [bone, current] of this.smoothed) {
      const target = this.rot.get(bone) || IDENTITY;
      current.slerp(target, k);
      const boneNode = humanoid.getNormalizedBoneNode(bone as VRMHumanBoneName);
      if (boneNode) {
        if (flip) {
          _scratchQuat.set(-current.x, current.y, -current.z, current.w);
          boneNode.quaternion.copy(_scratchQuat);
        } else {
          boneNode.quaternion.copy(current);
        }
      }
    }

    for (const [bone, target] of this.pos) {
      if (!this.smoothedPos.has(bone)) {
        this.smoothedPos.set(bone, [target[0], target[1], target[2]]);
      }
    }
    for (const [bone, current] of this.smoothedPos) {
      const target = this.pos.get(bone) || _zeroPos;
      current[0] += (target[0] - current[0]) * k;
      current[1] += (target[1] - current[1]) * k;
      current[2] += (target[2] - current[2]) * k;
      const boneNode = humanoid.getNormalizedBoneNode(bone as VRMHumanBoneName);
      if (boneNode) {
        if (flip) {
          boneNode.position.set(-current[0], current[1], -current[2]);
        } else {
          boneNode.position.set(current[0], current[1], current[2]);
        }
      }
    }
  }
}
