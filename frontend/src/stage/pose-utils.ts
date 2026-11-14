import * as THREE from 'three';
import type { VRM, VRMHumanBoneName } from '@pixiv/three-vrm';

const _e = new THREE.Euler();
const _qa = new THREE.Quaternion();
const _qb = new THREE.Quaternion();
const _va = new THREE.Vector3();
const _vb = new THREE.Vector3();
const _vc = new THREE.Vector3();
const _scratchQuat = new THREE.Quaternion();
const _zeroPos: [number, number, number] = [0, 0, 0];
const IDENTITY = new THREE.Quaternion();

// Preallocated scratch vectors and quaternions for Inverse Kinematics & Scapulohumeral Rhythm
const _ikTargetDir = new THREE.Vector3();
const _ikPole = new THREE.Vector3();
const _ikPolePerp = new THREE.Vector3();
const _ikElbow = new THREE.Vector3();
const _ikUpperDir = new THREE.Vector3();
const _ikBendDir = new THREE.Vector3();
const _ikCurZ = new THREE.Vector3();
const _ikCrossZ = new THREE.Vector3();
const _ikTwistQ = new THREE.Quaternion();
const _vArmRest = new THREE.Vector3();
const _vArmCurrent = new THREE.Vector3();

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
 * Analytical 2-Bone Arm Inverse Kinematics (IK) Solver.
 * Computes upperArm and lowerArm quaternions in VRM humanoid space to reach any 3D spatial target
 * with a natural physiological elbow swivel plane.
 * Guarantees zero heap allocations when outUpper and outLower are supplied.
 */
export function solveTwoBoneArmIK(
  side: Side,
  target: THREE.Vector3,
  outUpper = new THREE.Quaternion(),
  outLower = new THREE.Quaternion(),
  l1 = 0.25,
  l2 = 0.22,
  pole?: THREE.Vector3
): { upper: THREE.Quaternion; lower: THREE.Quaternion } {
  const s = sideSign(side);
  const d = target.length();
  const dMin = Math.abs(l1 - l2) + 0.005;
  const dMax = (l1 + l2) * 0.999;
  const dEff = clamp(d, dMin, dMax);

  if (d > 1e-6) {
    _ikTargetDir.copy(target).multiplyScalar(1 / d);
  } else {
    _ikTargetDir.set(0, -1, 0);
  }

  // Elbow flexion angle beta via law of cosines
  const cosElbow = clamp((l1 * l1 + l2 * l2 - dEff * dEff) / (2 * l1 * l2), -1, 1);
  const beta = Math.PI - Math.acos(cosElbow);

  // Shoulder angle gamma
  const cosGamma = clamp((l1 * l1 + dEff * dEff - l2 * l2) / (2 * l1 * dEff), -1, 1);
  const gamma = Math.acos(cosGamma);

  // Swivel pole vector
  if (pole) {
    _ikPole.copy(pole);
  } else {
    _ikPole.set(s * 0.2, -0.9, -0.2);
  }

  // Perpendicular component of pole relative to target line
  const proj = _ikPole.dot(_ikTargetDir);
  _ikPolePerp.copy(_ikPole).addScaledVector(_ikTargetDir, -proj);
  if (_ikPolePerp.lengthSq() < 1e-6) {
    _ikPolePerp.set(0, 0, 1).cross(_ikTargetDir);
    if (_ikPolePerp.lengthSq() < 1e-6) {
      _ikPolePerp.set(1, 0, 0).cross(_ikTargetDir);
    }
  }
  _ikPolePerp.normalize();

  // Elbow position vector in shoulder space
  _ikElbow.copy(_ikTargetDir).multiplyScalar(Math.cos(gamma) * l1)
    .addScaledVector(_ikPolePerp, Math.sin(gamma) * l1);

  _ikUpperDir.copy(_ikElbow).normalize();

  // Upper arm quaternion: rotate from rest vector (s, 0, 0) to _ikUpperDir
  _va.set(s, 0, 0);
  outUpper.setFromUnitVectors(_va, _ikUpperDir);

  // Bend direction in shoulder space
  _ikBendDir.copy(_ikTargetDir).addScaledVector(_ikUpperDir, -_ikTargetDir.dot(_ikUpperDir));
  if (_ikBendDir.lengthSq() < 1e-6) {
    _ikBendDir.copy(_ikPolePerp).negate();
  } else {
    _ikBendDir.normalize();
  }

  // Current local Z axis of upper arm transformed to shoulder space
  _ikCurZ.set(0, 0, 1).applyQuaternion(outUpper);

  const dotZ = clamp(_ikCurZ.dot(_ikBendDir), -1, 1);
  _ikCrossZ.crossVectors(_ikCurZ, _ikBendDir);
  const twistSign = _ikCrossZ.dot(_ikUpperDir) >= 0 ? 1 : -1;
  const twist = Math.acos(dotZ) * twistSign;

  if (Math.abs(twist) > 1e-4) {
    _ikTwistQ.setFromAxisAngle(_ikUpperDir, twist);
    outUpper.premultiply(_ikTwistQ);
  }

  // Lower arm (forearm) bends around Y axis by -s * beta
  _e.set(0, -s * beta, 0, 'XYZ');
  outLower.setFromEuler(_e);

  return { upper: outUpper, lower: outLower };
}

export interface SpatialAnchorDef {
  x: number;
  y: number;
  z: number;
  poleX: number;
  poleY: number;
  poleZ: number;
  fingers?: FingerCurls;
}

/**
 * Spatial Anatomical Anchors for Natural Human Actions.
 * Coordinates in shoulder-local space.
 */
export const SPATIAL_ANCHORS: Record<string, (side: Side) => SpatialAnchorDef> = {
  chin: (side) => {
    const s = sideSign(side);
    return {
      x: -s * 0.10,
      y: 0.08,
      z: 0.17,
      poleX: s * 0.3,
      poleY: -0.85,
      poleZ: -0.2,
      fingers: { all: 0.75, index: 0.3, thumb: 0.4 },
    };
  },
  cheek: (side) => {
    const s = sideSign(side);
    return {
      x: s * 0.05,
      y: 0.16,
      z: 0.15,
      poleX: s * 0.5,
      poleY: -0.7,
      poleZ: -0.1,
      fingers: { all: 0.25, thumb: 0.2 },
    };
  },
  mouth: (side) => {
    const s = sideSign(side);
    return {
      x: -s * 0.06,
      y: 0.10,
      z: 0.18,
      poleX: s * 0.2,
      poleY: -0.9,
      poleZ: -0.1,
      fingers: { all: 0.2, thumb: 0.15 },
    };
  },
  chestHeart: (side) => {
    const s = sideSign(side);
    return {
      x: side === 'right' ? 0.08 : 0.06,
      y: -0.12,
      z: 0.17,
      poleX: s * 0.2,
      poleY: -0.95,
      poleZ: -0.1,
      fingers: { all: 0.15, thumb: 0.1 },
    };
  },
  chestCenter: (side) => {
    const s = sideSign(side);
    return {
      x: -s * 0.02,
      y: -0.14,
      z: 0.18,
      poleX: s * 0.3,
      poleY: -0.9,
      poleZ: -0.1,
      fingers: { all: 0.08, thumb: 0.08 },
    };
  },
  forehead: (side) => {
    const s = sideSign(side);
    return {
      x: s * 0.07,
      y: 0.26,
      z: 0.14,
      poleX: s * 0.6,
      poleY: -0.6,
      poleZ: -0.2,
      fingers: { all: 0.85, index: 0.15, thumb: 0.4 },
    };
  },
  temple: (side) => {
    const s = sideSign(side);
    return {
      x: s * 0.11,
      y: 0.23,
      z: 0.11,
      poleX: s * 0.7,
      poleY: -0.5,
      poleZ: -0.1,
      fingers: { all: 0.8, index: 0.2, middle: 0.2, thumb: 0.35 },
    };
  },
  openEmbrace: (side) => {
    const s = sideSign(side);
    return {
      x: s * 0.22,
      y: -0.06,
      z: 0.36,
      poleX: s * 0.4,
      poleY: -0.8,
      poleZ: -0.3,
      fingers: { all: 0.15, thumb: 0.15 },
    };
  },
  lap: (side) => {
    const s = sideSign(side);
    return {
      x: s * 0.10,
      y: -0.36,
      z: 0.22,
      poleX: s * 0.2,
      poleY: -0.95,
      poleZ: -0.1,
      fingers: { all: 0.3, thumb: 0.2 },
    };
  },
};

export function getSpatialAnchor(
  name: string,
  side: Side,
  outTarget: THREE.Vector3,
  outPole: THREE.Vector3
): FingerCurls | undefined {
  const factory = SPATIAL_ANCHORS[name];
  if (!factory) return undefined;
  const def = factory(side);
  outTarget.set(def.x, def.y, def.z);
  outPole.set(def.poleX, def.poleY, def.poleZ);
  return def.fingers;
}

/**
 * Couples clavicle (shoulder) elevation and retraction to upper arm elevation.
 * Ratio: approx 1 degree clavicle per 3 degrees upper arm abduction/flexion (1:3 ratio).
 * Eliminates broken/detached shoulder syndrome during arm gestures.
 */
export function applyScapulohumeralRhythm(buf: PoseBuffer): void {
  for (let i = 0; i < 2; i++) {
    const side: Side = i === 0 ? 'left' : 'right';
    const s = sideSign(side);
    const armQuat = buf.get(`${side}UpperArm`);

    // Direction of upper arm from rest T-pose vector (s, 0, 0)
    _vArmRest.set(s, 0, 0);
    _vArmCurrent.copy(_vArmRest).applyQuaternion(armQuat);

    // Angle relative to vertical downward direction (0, -1, 0)
    const cosAngleFromDown = clamp(-_vArmCurrent.y, -1, 1);
    const angleFromDown = Math.acos(cosAngleFromDown);

    // Resting hang angle is ~0.34 rad. Elevation above rest:
    const elev = Math.max(0, angleFromDown - 0.34);
    if (elev < 1e-4) continue;

    // Lateral abduction component (in coronal plane)
    const abd = Math.max(0, Math.atan2(Math.max(0, _vArmCurrent.x * s), Math.max(0.01, -_vArmCurrent.y)) - 0.32);

    // Forward flexion component (in sagittal plane)
    const flex = Math.max(0, Math.atan2(_vArmCurrent.z, Math.max(0.01, -_vArmCurrent.y)) - 0.14);

    // 1:3 ratio (0.333) for clavicle elevation
    const clavicleElevation = (abd + 0.75 * flex) * 0.333;

    // Clavicle retraction (backward) / protraction (forward)
    const clavicleRetraction = (abd * 0.25 - flex * 0.35) * 0.333;

    // Clavicle axial roll
    const clavicleRoll = (flex * 0.2) * 0.333;

    // Add to shoulder bone:
    // Left: +Z is elevation, +Y is retraction.
    // Right: -Z is elevation, -Y is retraction.
    buf.add(`${side}Shoulder`, clavicleRoll, s * clavicleRetraction, s * clavicleElevation);
  }
}

/**
 * Maps a unit quaternion q in SO(3) to its Lie algebra vector in so(3) ~= R^3.
 * theta_vec = Log(q)
 * Uses Padé/Taylor polynomial small-angle approximation for theta < 0.4 rad (~23°),
 * falling back to exact trig (atan2) for larger angles.
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
  const s = x * x + y * y + z * z;
  let factor: number;
  if (s < 1e-12) {
    factor = 2.0;
  } else if (s < 0.04) {
    // Padé / Taylor series for (2 * arcsin(sqrt(s))) / sqrt(s)
    // 2 * (1 + (1/6)*s + (3/40)*s^2 + (5/112)*s^3)
    // For theta < 0.4 rad (~23 deg), absolute difference is < 1.6e-7 rad (< 32-bit float epsilon)
    // Completely eliminates Math.atan2, Math.hypot, and division.
    factor = 2.0 * (1.0 + s * (0.16666666666666666 + s * (0.075 + s * 0.04464285714285714)));
  } else {
    const vLen = Math.sqrt(s);
    factor = (2.0 * Math.atan2(vLen, w)) / vLen;
  }
  return out.set(x * factor, y * factor, z * factor);
}

/**
 * Maps a Lie algebra vector in so(3) ~= R^3 to its unit quaternion in SO(3).
 * q = Exp(v)
 * Uses Taylor polynomial small-angle approximation for theta < 0.2 rad,
 * falling back to exact trig for larger angles.
 */
export function so3Exp(v: THREE.Vector3, out = new THREE.Quaternion()): THREE.Quaternion {
  const thetaSq = v.x * v.x + v.y * v.y + v.z * v.z;
  if (thetaSq < 1e-12) {
    return out.set(0, 0, 0, 1);
  }
  let factor: number;
  let w: number;
  if (thetaSq < 0.04) {
    factor = 0.5 - thetaSq * 0.020833333333333332;
    w = 1.0 - thetaSq * 0.125;
  } else {
    const theta = Math.sqrt(thetaSq);
    const half = theta * 0.5;
    factor = Math.sin(half) / theta;
    w = Math.cos(half);
  }
  return out.set(v.x * factor, v.y * factor, v.z * factor, w).normalize();
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

/**
 * Packed Struct-of-Arrays (SoA) Vectorized Spring Damper Batch on SO(3).
 * 16 floats (64 bytes = 1 L1 cache line) per joint.
 */
export class VectorizedSpringDamperBatch {
  public readonly count: number;
  public readonly buffer: Float32Array;

  constructor(count: number) {
    this.count = count;
    this.buffer = new Float32Array(count * 16);
    for (let i = 0; i < count; i++) {
      const b = i * 16;
      this.buffer[b + 3] = 1.0;  // curr_q.w
      this.buffer[b + 11] = 1.0; // targ_q.w
      const omegaN = 14;
      const zeta = 0.85;
      this.buffer[b + 12] = omegaN * omegaN;      // kSpring
      this.buffer[b + 13] = 2.0 * zeta * omegaN; // kDamp
    }
  }

  public stepAll(dt: number): void {
    const buf = this.buffer;
    const count = this.count;
    const clampedDt = dt > 0.05 ? 0.05 : dt;

    for (let i = 0; i < count; i++) {
      const b = i * 16;
      let cx = buf[b], cy = buf[b + 1], cz = buf[b + 2], cw = buf[b + 3];
      const tx = buf[b + 8], ty = buf[b + 9], tz = buf[b + 10], tw = buf[b + 11];

      // q_rel = q_curr^-1 * q_targ = (-cx, -cy, -cz, cw) * (tx, ty, tz, tw)
      let rx =  cw * tx - cz * ty + cy * tz - cx * tw;
      let ry =  cz * tx + cw * ty - cx * tz - cy * tw;
      let rz = -cy * tx + cx * ty + cw * tz - cz * tw;
      let rw =  cx * tx + cy * ty + cz * tz + cw * tw;

      if (rw < 0) { rx = -rx; ry = -ry; rz = -rz; rw = -rw; }

      // Polynomial so3Log
      const s = rx * rx + ry * ry + rz * rz;
      let logFactor: number;
      if (s < 1e-12) {
        logFactor = 2.0;
      } else if (s < 0.04) {
        logFactor = 2.0 * (1.0 + s * (0.16666666666666666 + s * (0.075 + s * 0.04464285714285714)));
      } else {
        const vLen = Math.sqrt(s);
        logFactor = (2.0 * Math.atan2(vLen, rw)) / vLen;
      }

      const ex = rx * logFactor;
      const ey = ry * logFactor;
      const ez = rz * logFactor;

      // Dynamics
      const kSpring = buf[b + 12];
      const kDamp = buf[b + 13];
      let vx = buf[b + 4] + (ex * kSpring - buf[b + 4] * kDamp) * clampedDt;
      let vy = buf[b + 5] + (ey * kSpring - buf[b + 5] * kDamp) * clampedDt;
      let vz = buf[b + 6] + (ez * kSpring - buf[b + 6] * kDamp) * clampedDt;

      buf[b + 4] = vx; buf[b + 5] = vy; buf[b + 6] = vz;

      // Polynomial so3Exp
      const drx = vx * clampedDt, dry = vy * clampedDt, drz = vz * clampedDt;
      const thetaSq = drx * drx + dry * dry + drz * drz;
      let expFactor: number, expW: number;
      if (thetaSq < 1e-12) {
        expFactor = 0.5; expW = 1.0;
      } else if (thetaSq < 0.04) {
        expFactor = 0.5 - thetaSq * 0.020833333333333332;
        expW = 1.0 - thetaSq * 0.125;
      } else {
        const theta = Math.sqrt(thetaSq);
        const half = theta * 0.5;
        expFactor = Math.sin(half) / theta;
        expW = Math.cos(half);
      }

      const dqx = drx * expFactor, dqy = dry * expFactor, dqz = drz * expFactor, dqw = expW;

      // q_new = q_curr * dq
      let nx = cw * dqx + cx * dqw + cy * dqz - cz * dqy;
      let ny = cw * dqy - cx * dqz + cy * dqw + cz * dqx;
      let nz = cw * dqz + cx * dqy - cy * dqx + cz * dqw;
      let nw = cw * dqw - cx * dqx - cy * dqy - cz * dqz;

      const normInv = 1.0 / Math.sqrt(nx * nx + ny * ny + nz * nz + nw * nw);
      buf[b] = nx * normInv;
      buf[b + 1] = ny * normInv;
      buf[b + 2] = nz * normInv;
      buf[b + 3] = nw * normInv;
    }
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

export const MAX_BONES = 128;

export const BONE_NAMES = [
  'hips',
  'spine',
  'chest',
  'upperChest',
  'neck',
  'head',
  'leftEye',
  'rightEye',
  'jaw',
  'leftShoulder',
  'leftUpperArm',
  'leftLowerArm',
  'leftHand',
  'rightShoulder',
  'rightUpperArm',
  'rightLowerArm',
  'rightHand',
  'leftUpperLeg',
  'leftLowerLeg',
  'leftFoot',
  'leftToes',
  'rightUpperLeg',
  'rightLowerLeg',
  'rightFoot',
  'rightToes',
  // Left fingers
  'leftThumbProximal',
  'leftThumbIntermediate',
  'leftThumbDistal',
  'leftIndexProximal',
  'leftIndexIntermediate',
  'leftIndexDistal',
  'leftMiddleProximal',
  'leftMiddleIntermediate',
  'leftMiddleDistal',
  'leftRingProximal',
  'leftRingIntermediate',
  'leftRingDistal',
  'leftLittleProximal',
  'leftLittleIntermediate',
  'leftLittleDistal',
  // Right fingers
  'rightThumbProximal',
  'rightThumbIntermediate',
  'rightThumbDistal',
  'rightIndexProximal',
  'rightIndexIntermediate',
  'rightIndexDistal',
  'rightMiddleProximal',
  'rightMiddleIntermediate',
  'rightMiddleDistal',
  'rightRingProximal',
  'rightRingIntermediate',
  'rightRingDistal',
  'rightLittleProximal',
  'rightLittleIntermediate',
  'rightLittleDistal',
] as const;

export class PoseBuffer {
  public rot = new Map<string, THREE.Quaternion>();
  public pos = new Map<string, [number, number, number]>();
  public smoothed = new Map<string, THREE.Quaternion>();
  public smoothedPos = new Map<string, [number, number, number]>();

  // Preallocated contiguous Float32Array buffers for zero-allocation 60fps render loop
  public readonly rotBuffer = new Float32Array(MAX_BONES * 4);
  public readonly posBuffer = new Float32Array(MAX_BONES * 3);
  public readonly smoothedRotBuffer = new Float32Array(MAX_BONES * 4);
  public readonly smoothedPosBuffer = new Float32Array(MAX_BONES * 3);

  private _boneNameToId = new Map<string, number>();
  private _idToBoneName: string[] = [];
  private _boneCount = 0;

  private _rotObjects: THREE.Quaternion[] = [];
  private _posObjects: [number, number, number][] = [];
  private _smoothedRotObjects: THREE.Quaternion[] = [];
  private _smoothedPosObjects: [number, number, number][] = [];

  private _activeRotFlags = new Uint8Array(MAX_BONES);
  private _activeRotIds = new Int32Array(MAX_BONES);
  private _activeRotCount = 0;

  private _activePosFlags = new Uint8Array(MAX_BONES);
  private _activePosIds = new Int32Array(MAX_BONES);
  private _activePosCount = 0;

  private _hasSmoothedRot = new Uint8Array(MAX_BONES);
  private _allSmoothedRotIds = new Int32Array(MAX_BONES);
  private _allSmoothedRotCount = 0;

  private _hasSmoothedPos = new Uint8Array(MAX_BONES);
  private _allSmoothedPosIds = new Int32Array(MAX_BONES);
  private _allSmoothedPosCount = 0;

  private _boneNodeCache: (THREE.Object3D | null | undefined)[] = new Array(MAX_BONES);
  private _nodeCacheHumanoid: any = null;

  constructor() {
    for (let i = 0; i < MAX_BONES; i++) {
      this._rotObjects[i] = new THREE.Quaternion();
      this._posObjects[i] = [0, 0, 0];
      this._smoothedRotObjects[i] = new THREE.Quaternion();
      this._smoothedPosObjects[i] = [0, 0, 0];
    }
    for (let i = 0; i < BONE_NAMES.length; i++) {
      this._registerBone(BONE_NAMES[i]);
    }
  }

  private _registerBone(name: string): number {
    let id = this._boneNameToId.get(name);
    if (id !== undefined) return id;
    if (this._boneCount >= MAX_BONES) {
      return 0;
    }
    id = this._boneCount++;
    this._boneNameToId.set(name, id);
    this._idToBoneName[id] = name;
    return id;
  }

  private _getBoneId(name: string): number {
    let id = this._boneNameToId.get(name);
    if (id === undefined) {
      id = this._registerBone(name);
    }
    return id;
  }

  private _getCachedBoneNode(humanoid: any, id: number, boneName: string): any {
    if (this._nodeCacheHumanoid !== humanoid) {
      this._nodeCacheHumanoid = humanoid;
      this._boneNodeCache.fill(undefined);
    }
    let node = this._boneNodeCache[id];
    if (node === undefined) {
      node = humanoid.getNormalizedBoneNode?.(boneName as VRMHumanBoneName) ?? null;
      this._boneNodeCache[id] = node;
    }
    return node;
  }

  public reset(): void {
    this.rot.clear();
    this.pos.clear();
    this._activeRotFlags.fill(0);
    this._activePosFlags.fill(0);
    this._activeRotCount = 0;
    this._activePosCount = 0;
  }

  public clearSmoothing(): void {
    this.smoothed.clear();
    this.smoothedPos.clear();
    this._hasSmoothedRot.fill(0);
    this._hasSmoothedPos.fill(0);
    this._allSmoothedRotCount = 0;
    this._allSmoothedPosCount = 0;
    this._nodeCacheHumanoid = null;
    this._boneNodeCache.fill(undefined);
  }

  public get(bone: string): THREE.Quaternion {
    const id = this._getBoneId(bone);
    const q = this._rotObjects[id];
    if (!this._activeRotFlags[id]) {
      this._activeRotFlags[id] = 1;
      this._activeRotIds[this._activeRotCount++] = id;
      q.identity();
      this.rot.set(bone, q);
      const b = id * 4;
      this.rotBuffer[b] = 0;
      this.rotBuffer[b + 1] = 0;
      this.rotBuffer[b + 2] = 0;
      this.rotBuffer[b + 3] = 1;
    }
    return q;
  }

  public set(bone: string, q: THREE.Quaternion): void {
    const target = this.get(bone);
    target.copy(q);
    const id = this._getBoneId(bone);
    const b = id * 4;
    this.rotBuffer[b] = target.x;
    this.rotBuffer[b + 1] = target.y;
    this.rotBuffer[b + 2] = target.z;
    this.rotBuffer[b + 3] = target.w;
  }

  public add(bone: string, x: number, y: number, z: number): void {
    const target = this.get(bone);
    target.multiply(eq(x, y, z, _qb));
    const id = this._getBoneId(bone);
    const b = id * 4;
    this.rotBuffer[b] = target.x;
    this.rotBuffer[b + 1] = target.y;
    this.rotBuffer[b + 2] = target.z;
    this.rotBuffer[b + 3] = target.w;
  }

  public blend(bone: string, q: THREE.Quaternion, w: number): void {
    if (w <= 0) return;
    const target = this.get(bone);
    target.slerp(q, clamp(w, 0, 1));
    const id = this._getBoneId(bone);
    const b = id * 4;
    this.rotBuffer[b] = target.x;
    this.rotBuffer[b + 1] = target.y;
    this.rotBuffer[b + 2] = target.z;
    this.rotBuffer[b + 3] = target.w;
  }

  public offsetPos(bone: string, x: number, y: number, z: number): void {
    const id = this._getBoneId(bone);
    const p = this._posObjects[id];
    if (!this._activePosFlags[id]) {
      this._activePosFlags[id] = 1;
      this._activePosIds[this._activePosCount++] = id;
      p[0] = 0;
      p[1] = 0;
      p[2] = 0;
      this.pos.set(bone, p);
    }
    p[0] += x;
    p[1] += y;
    p[2] += z;
    const b = id * 3;
    this.posBuffer[b] = p[0];
    this.posBuffer[b + 1] = p[1];
    this.posBuffer[b + 2] = p[2];
  }

  public setFingers(side: Side, curls: FingerCurls): void {
    const s = sideSign(side);
    const prefix = side;
    for (let f = 0; f < FINGERS.length; f++) {
      const name = FINGERS[f];
      const key = name.toLowerCase() as keyof FingerCurls;
      const c = curls[key] ?? curls.all ?? 0;
      for (let j = 0; j < JOINTS.length; j++) {
        const boneName = `${prefix}${name}${JOINTS[j]}`;
        const target = this.get(boneName);
        eq(0, 0, -s * c * GAINS[j], target);
        const id = this._getBoneId(boneName);
        const b = id * 4;
        this.rotBuffer[b] = target.x;
        this.rotBuffer[b + 1] = target.y;
        this.rotBuffer[b + 2] = target.z;
        this.rotBuffer[b + 3] = target.w;
      }
    }
    const t = curls.thumb ?? (curls.all ?? 0) * 0.5;
    const thumbProx = `${prefix}ThumbProximal`;
    const qProx = this.get(thumbProx);
    eq(0, s * t * 0.35, -s * t * 0.2, qProx);
    let id = this._getBoneId(thumbProx);
    let b = id * 4;
    this.rotBuffer[b] = qProx.x;
    this.rotBuffer[b + 1] = qProx.y;
    this.rotBuffer[b + 2] = qProx.z;
    this.rotBuffer[b + 3] = qProx.w;

    const thumbDist = `${prefix}ThumbDistal`;
    const qDist = this.get(thumbDist);
    eq(0, 0, -s * t * 0.5, qDist);
    id = this._getBoneId(thumbDist);
    b = id * 4;
    this.rotBuffer[b] = qDist.x;
    this.rotBuffer[b + 1] = qDist.y;
    this.rotBuffer[b + 2] = qDist.z;
    this.rotBuffer[b + 3] = qDist.w;
  }

  public apply(vrm: VRM, dt = 0.016, smoothing = 14): void {
    const humanoid = vrm?.humanoid;
    if (!humanoid) return;

    const flip = vrm.meta?.metaVersion === '0';
    const safeDt = typeof dt === 'number' && !isNaN(dt) ? Math.max(dt, 0) : 0.016;
    const k = 1 - Math.exp(-smoothing * safeDt);

    // Sync from Map if external caller modified rot map directly
    if (this.rot.size > this._activeRotCount) {
      for (const bone of this.rot.keys()) {
        const id = this._getBoneId(bone);
        if (!this._activeRotFlags[id]) {
          this._activeRotFlags[id] = 1;
          this._activeRotIds[this._activeRotCount++] = id;
          const q = this.rot.get(bone)!;
          this._rotObjects[id].copy(q);
        }
      }
    }

    // 1. Sync rotBuffer and initialize newly active smoothed bones
    for (let i = 0; i < this._activeRotCount; i++) {
      const id = this._activeRotIds[i];
      const q = this._rotObjects[id];
      const b4 = id * 4;
      this.rotBuffer[b4] = q.x;
      this.rotBuffer[b4 + 1] = q.y;
      this.rotBuffer[b4 + 2] = q.z;
      this.rotBuffer[b4 + 3] = q.w;

      if (!this._hasSmoothedRot[id]) {
        this._hasSmoothedRot[id] = 1;
        this._allSmoothedRotIds[this._allSmoothedRotCount++] = id;
        this.smoothedRotBuffer[b4] = q.x;
        this.smoothedRotBuffer[b4 + 1] = q.y;
        this.smoothedRotBuffer[b4 + 2] = q.z;
        this.smoothedRotBuffer[b4 + 3] = q.w;
        this._smoothedRotObjects[id].copy(q);
        this.smoothed.set(this._idToBoneName[id], this._smoothedRotObjects[id]);
      }
    }

    // 2. Step smoothed rotations: indexed loop eliminates iterator and [k, v] tuple allocations
    for (let i = 0; i < this._allSmoothedRotCount; i++) {
      const id = this._allSmoothedRotIds[i];
      const boneName = this._idToBoneName[id];
      const current = this._smoothedRotObjects[id];
      const target = this._activeRotFlags[id] ? this._rotObjects[id] : IDENTITY;
      current.slerp(target, k);

      const b4 = id * 4;
      this.smoothedRotBuffer[b4] = current.x;
      this.smoothedRotBuffer[b4 + 1] = current.y;
      this.smoothedRotBuffer[b4 + 2] = current.z;
      this.smoothedRotBuffer[b4 + 3] = current.w;

      const boneNode = this._getCachedBoneNode(humanoid, id, boneName);
      if (boneNode) {
        if (flip) {
          _scratchQuat.set(-current.x, current.y, -current.z, current.w);
          boneNode.quaternion.copy(_scratchQuat);
        } else {
          boneNode.quaternion.copy(current);
        }
      }
    }

    // Sync from Map if external caller modified pos map directly
    if (this.pos.size > this._activePosCount) {
      for (const bone of this.pos.keys()) {
        const id = this._getBoneId(bone);
        if (!this._activePosFlags[id]) {
          this._activePosFlags[id] = 1;
          this._activePosIds[this._activePosCount++] = id;
          const p = this.pos.get(bone)!;
          const po = this._posObjects[id];
          po[0] = p[0];
          po[1] = p[1];
          po[2] = p[2];
        }
      }
    }

    // 3. Sync posBuffer and initialize newly active smoothed positions
    for (let i = 0; i < this._activePosCount; i++) {
      const id = this._activePosIds[i];
      const p = this._posObjects[id];
      const b3 = id * 3;
      this.posBuffer[b3] = p[0];
      this.posBuffer[b3 + 1] = p[1];
      this.posBuffer[b3 + 2] = p[2];

      if (!this._hasSmoothedPos[id]) {
        this._hasSmoothedPos[id] = 1;
        this._allSmoothedPosIds[this._allSmoothedPosCount++] = id;
        this.smoothedPosBuffer[b3] = p[0];
        this.smoothedPosBuffer[b3 + 1] = p[1];
        this.smoothedPosBuffer[b3 + 2] = p[2];
        const sp = this._smoothedPosObjects[id];
        sp[0] = p[0];
        sp[1] = p[1];
        sp[2] = p[2];
        this.smoothedPos.set(this._idToBoneName[id], sp);
      }
    }

    // 4. Step smoothed positions: indexed loop eliminates iterator and [k, v] tuple allocations
    for (let i = 0; i < this._allSmoothedPosCount; i++) {
      const id = this._allSmoothedPosIds[i];
      const boneName = this._idToBoneName[id];
      const current = this._smoothedPosObjects[id];
      const target = this._activePosFlags[id] ? this._posObjects[id] : _zeroPos;

      current[0] += (target[0] - current[0]) * k;
      current[1] += (target[1] - current[1]) * k;
      current[2] += (target[2] - current[2]) * k;

      const b3 = id * 3;
      this.smoothedPosBuffer[b3] = current[0];
      this.smoothedPosBuffer[b3 + 1] = current[1];
      this.smoothedPosBuffer[b3 + 2] = current[2];

      const boneNode = this._getCachedBoneNode(humanoid, id, boneName);
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
