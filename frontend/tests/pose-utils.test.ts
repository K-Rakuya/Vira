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
} from '../src/stage/pose-utils';

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
});
