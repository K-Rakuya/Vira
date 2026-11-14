import { describe, expect, it } from 'bun:test';
import * as THREE from 'three';
import {
  clamp,
  lerp,
  smoothstep,
  noise,
  eq,
  upperArmQuat,
  lowerArmQuat,
  PoseBuffer,
  so3Log,
  so3Exp,
  GeodesicSpringDamper,
  VectorizedSpringDamperBatch,
} from '../src/stage/pose-utils';
import {
  so3Log as rendererSo3Log,
  PoseBuffer as RendererPoseBuffer,
} from '../src/renderer/pose-utils';
import { SceneManager, VRMViewer } from '../src/renderer/scene-manager';
import { parseKinematicCues } from '../src/stage/procedural-animator';

describe('pose-utils math helpers', () => {
  it('clamp correctly bounds numbers', () => {
    expect(clamp(5, 0, 10)).toBe(5);
    expect(clamp(-5, 0, 10)).toBe(0);
    expect(clamp(15, 0, 10)).toBe(10);
  });

  it('lerp performs linear interpolation', () => {
    expect(lerp(0, 100, 0.5)).toBe(50);
    expect(lerp(10, 20, 0)).toBe(10);
    expect(lerp(10, 20, 1)).toBe(20);
  });

  it('smoothstep creates S-curve transitions', () => {
    expect(smoothstep(0)).toBe(0);
    expect(smoothstep(1)).toBe(1);
    expect(smoothstep(0.5)).toBe(0.5);
    expect(smoothstep(-0.5)).toBe(0);
    expect(smoothstep(1.5)).toBe(1);
  });

  it('noise returns pseudo-random wave bounded within [-1, 1]', () => {
    for (let t = 0; t < 10; t += 0.5) {
      const val = noise(t, 42);
      expect(val).toBeGreaterThanOrEqual(-1);
      expect(val).toBeLessThanOrEqual(1);
    }
  });

  it('eq generates quaternion from euler angles', () => {
    const q = eq(0, Math.PI / 2, 0);
    expect(q).toBeInstanceOf(THREE.Quaternion);
    expect(q.length()).toBeCloseTo(1.0, 4);
  });

  it('upperArmQuat and lowerArmQuat produce unit quaternions for both sides', () => {
    const leftArm = upperArmQuat('left', 0.5, 0.2, 0.1);
    const rightArm = upperArmQuat('right', 0.5, 0.2, 0.1);
    const leftForearm = lowerArmQuat('left', 0.4);
    const rightForearm = lowerArmQuat('right', 0.4);

    expect(leftArm.length()).toBeCloseTo(1.0, 4);
    expect(rightArm.length()).toBeCloseTo(1.0, 4);
    expect(leftForearm.length()).toBeCloseTo(1.0, 4);
    expect(rightForearm.length()).toBeCloseTo(1.0, 4);
  });
});

describe('SO(3) Lie Group / Lie Algebra operations', () => {
  it('so3Log maps identity quaternion to zero vector', () => {
    const qId = new THREE.Quaternion(0, 0, 0, 1);
    const v = so3Log(qId);
    expect(v.length()).toBeCloseTo(0, 5);
  });

  it('so3Exp maps zero vector to identity quaternion', () => {
    const vZero = new THREE.Vector3(0, 0, 0);
    const q = so3Exp(vZero);
    expect(q.x).toBeCloseTo(0, 5);
    expect(q.y).toBeCloseTo(0, 5);
    expect(q.z).toBeCloseTo(0, 5);
    expect(q.w).toBeCloseTo(1, 5);
  });

  it('so3Log and so3Exp are invertible roundtrips for general 3D rotations', () => {
    const original = new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(1, 2, 3).normalize(),
      0.75
    );
    const tangent = so3Log(original);
    const restored = so3Exp(tangent);

    // Quaternions q and -q represent the exact same rotation
    const dot = Math.abs(original.dot(restored));
    expect(dot).toBeCloseTo(1.0, 4);
  });

  it('GeodesicSpringDamper drives current quaternion toward target smoothly', () => {
    const damper = new GeodesicSpringDamper(12.0, 1.0);
    const target = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), 0.5);

    // Initial state is identity
    expect(damper.current.y).toBeCloseTo(0, 4);

    // Step forward over time
    for (let i = 0; i < 30; i++) {
      damper.step(target, 1 / 60);
    }

    // After 0.5s with omega_n=12, critically damped spring should have converged close to target
    const dot = Math.abs(damper.current.dot(target));
    expect(dot).toBeGreaterThan(0.95);
  });

  it('so3Log uses Padé/Taylor approximation for theta < 0.4 rad and exact trig for theta >= 0.4 rad', () => {
    // Exact trigonometric reference function
    const exactLog = (q: THREE.Quaternion) => {
      let w = clamp(q.w, -1, 1);
      let x = q.x, y = q.y, z = q.z;
      if (w < 0) { w = -w; x = -x; y = -y; z = -z; }
      const vLen = Math.hypot(x, y, z);
      if (vLen < 1e-7) return new THREE.Vector3(2 * x, 2 * y, 2 * z);
      const factor = (2 * Math.atan2(vLen, w)) / vLen;
      return new THREE.Vector3(x * factor, y * factor, z * factor);
    };

    const axis = new THREE.Vector3(1, 2, 3).normalize();

    // Small angles (< 0.4 rad) evaluated with polynomial
    for (const theta of [0.0001, 0.01, 0.05, 0.2, 0.35, 0.395]) {
      const q = new THREE.Quaternion().setFromAxisAngle(axis, theta);
      const approx = so3Log(q);
      const exact = exactLog(q);
      expect(Math.abs(approx.x - exact.x)).toBeLessThan(1e-6);
      expect(Math.abs(approx.y - exact.y)).toBeLessThan(1e-6);
      expect(Math.abs(approx.z - exact.z)).toBeLessThan(1e-6);
      expect(approx.length()).toBeCloseTo(theta, 5);
    }

    // Larger angles (>= 0.4 rad) falling back to exact trig
    for (const theta of [0.45, 0.8, 1.5, 2.8]) {
      const q = new THREE.Quaternion().setFromAxisAngle(axis, theta);
      const res = so3Log(q);
      const exact = exactLog(q);
      expect(Math.abs(res.x - exact.x)).toBeLessThan(1e-6);
      expect(Math.abs(res.y - exact.y)).toBeLessThan(1e-6);
      expect(Math.abs(res.z - exact.z)).toBeLessThan(1e-6);
      expect(res.length()).toBeCloseTo(theta, 4);
    }
  });

  it('VectorizedSpringDamperBatch integrates joints with Float32Array SoA', () => {
    const batch = new VectorizedSpringDamperBatch(2);
    expect(batch.count).toBe(2);
    expect(batch.buffer.length).toBe(32);

    // Target for joint 0: 0.3 rad around Y
    const targetQ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), 0.3);
    batch.buffer[8] = targetQ.x;
    batch.buffer[9] = targetQ.y;
    batch.buffer[10] = targetQ.z;
    batch.buffer[11] = targetQ.w;

    for (let step = 0; step < 60; step++) {
      batch.stepAll(1 / 60);
    }

    // Joint 0 should have converged toward target
    const currentQ = new THREE.Quaternion(batch.buffer[0], batch.buffer[1], batch.buffer[2], batch.buffer[3]);
    const dot = Math.abs(currentQ.dot(targetQ));
    expect(dot).toBeGreaterThan(0.98);
  });
});

describe('PoseBuffer', () => {
  it('records and blends bone rotations', () => {
    const buf = new PoseBuffer();
    expect(buf.rot.size).toBe(0);

    const q = eq(0.1, 0.2, 0.3);
    buf.set('head', q);
    expect(buf.get('head').equals(q)).toBe(true);

    buf.add('head', 0.1, 0, 0);
    expect(buf.get('head').equals(q)).toBe(false);

    buf.offsetPos('hips', 0, 0.05, 0);
    expect(buf.pos.get('hips')).toEqual([0, 0.05, 0]);

    buf.reset();
    expect(buf.rot.size).toBe(0);
    expect(buf.pos.size).toBe(0);
  });

  it('configures finger curls for both hands', () => {
    const buf = new PoseBuffer();
    buf.setFingers('right', { all: 0.5, index: 0.1 });
    expect(buf.rot.has('rightIndexProximal')).toBe(true);
    expect(buf.rot.has('rightMiddleProximal')).toBe(true);
    expect(buf.rot.has('rightThumbProximal')).toBe(true);
  });

  it('zero-alloc apply mutates bone nodes directly', () => {
    const buf = new PoseBuffer();
    const rot = eq(0.2, 0.1, -0.1);
    buf.set('head', rot);
    buf.offsetPos('hips', 0.01, 0.02, 0.03);

    const mockBoneNode = new THREE.Object3D();
    const mockHipsNode = new THREE.Object3D();

    const mockVRM = {
      humanoid: {
        getNormalizedBoneNode: (name: string) => {
          if (name === 'head') return mockBoneNode;
          if (name === 'hips') return mockHipsNode;
          return null;
        },
      },
    };

    buf.apply(mockVRM as any);
    expect(mockBoneNode.quaternion.x).toBeCloseTo(rot.x, 4);
    expect(mockBoneNode.quaternion.y).toBeCloseTo(rot.y, 4);
    expect(mockBoneNode.quaternion.z).toBeCloseTo(rot.z, 4);
    expect(mockBoneNode.quaternion.w).toBeCloseTo(rot.w, 4);

    expect(mockHipsNode.position.x).toBeCloseTo(0.01, 4);
    expect(mockHipsNode.position.y).toBeCloseTo(0.02, 4);
    expect(mockHipsNode.position.z).toBeCloseTo(0.03, 4);
  });

  it('preallocates Float32Array buffers for zero-allocation bone transforms', () => {
    const buf = new PoseBuffer();
    expect(buf.rotBuffer).toBeInstanceOf(Float32Array);
    expect(buf.posBuffer).toBeInstanceOf(Float32Array);
    expect(buf.smoothedRotBuffer).toBeInstanceOf(Float32Array);
    expect(buf.smoothedPosBuffer).toBeInstanceOf(Float32Array);

    const rot = eq(0.1, 0.2, 0.3);
    buf.set('head', rot);
    buf.offsetPos('hips', 0.05, 0.1, 0.15);

    // Check that rotBuffer contains head rotation (head index = 5)
    expect(buf.rotBuffer[5 * 4 + 0]).toBeCloseTo(rot.x, 5);
    expect(buf.rotBuffer[5 * 4 + 1]).toBeCloseTo(rot.y, 5);
    expect(buf.rotBuffer[5 * 4 + 2]).toBeCloseTo(rot.z, 5);
    expect(buf.rotBuffer[5 * 4 + 3]).toBeCloseTo(rot.w, 5);

    // Check that posBuffer contains hips translation (hips index = 0)
    expect(buf.posBuffer[0 * 3 + 0]).toBeCloseTo(0.05, 5);
    expect(buf.posBuffer[0 * 3 + 1]).toBeCloseTo(0.1, 5);
    expect(buf.posBuffer[0 * 3 + 2]).toBeCloseTo(0.15, 5);

    // Simulate 60fps render loop apply calls
    const mockVRM = {
      humanoid: {
        getNormalizedBoneNode: () => new THREE.Object3D(),
      },
    };
    for (let f = 0; f < 10; f++) {
      buf.apply(mockVRM as any, 1 / 60, 14);
    }

    // smoothed buffers should be updated
    expect(buf.smoothedRotBuffer[5 * 4 + 3]).toBeGreaterThan(0);
    expect(buf.smoothedPosBuffer[0 * 3 + 1]).toBeGreaterThan(0);
  });

  it('renderer module exports pose-utils and scene-manager seamlessly', () => {
    expect(rendererSo3Log).toBe(so3Log);
    const rBuf = new RendererPoseBuffer();
    expect(rBuf.rotBuffer).toBeInstanceOf(Float32Array);

    expect(SceneManager).toBe(VRMViewer);
    const viewer = new SceneManager();
    expect(viewer.boneRotBuffer).toBeInstanceOf(Float32Array);
    expect(viewer.bonePosBuffer).toBeInstanceOf(Float32Array);
  });
});

describe('Kinematic Cue Synthesizer', () => {
  it('parses Vietnamese implicit gestures and emotions', () => {
    const cue1 = parseKinematicCues('Chào bạn nhé! *gật đầu*');
    expect(cue1.gesture).toBe('nod');

    const cue2 = parseKinematicCues('Tôi rất vui được gặp bạn *mỉm cười*');
    expect(cue2.emotion).toBe('happy');

    const cue3 = parseKinematicCues('*cúi đầu* Xin kính chào quý khách');
    expect(cue3.gesture).toBe('bow');

    const cue4 = parseKinematicCues('Ồ, điều này thật lạ *nghiêng đầu*');
    expect(cue4.gesture).toBe('tilt');
  });

  it('parses English implicit gestures and explicit tags', () => {
    const cue1 = parseKinematicCues('Hello there! *waves*');
    expect(cue1.gesture).toBe('wave');

    const cue2 = parseKinematicCues('I see your point. [gesture:explain]');
    expect(cue2.gesture).toBe('explain');

    const cue3 = parseKinematicCues('Great job! *cheers*');
    expect(cue3.gesture).toBe('praise');
  });
});
