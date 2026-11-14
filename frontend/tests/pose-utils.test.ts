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
} from '../src/stage/pose-utils';
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
