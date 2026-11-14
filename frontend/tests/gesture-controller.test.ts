import { describe, expect, it } from 'bun:test';
import {
  GestureController,
  GESTURES,
  REST_CASCADE,
  OPEN_CASCADE,
  POINT_CASCADE,
  FIST_CASCADE,
  quintic,
} from '../src/stage/gesture-controller';
import { PoseBuffer } from '../src/stage/pose-utils';

describe('GestureController', () => {
  it('contains expected list of procedural gestures including tilt and shrug', () => {
    const ctrl = new GestureController();
    const expected = [
      'wave',
      'explain',
      'point',
      'think',
      'idea',
      'nod',
      'shake',
      'praise',
      'clap',
      'bow',
      'shy',
      'encourage',
      'tilt',
      'shrug',
      'surprise',
      'relax',
      'chin_touch',
      'mouth_cover',
      'hand_on_heart',
      'temple_rub',
      'embrace',
      'rub_hands',
      'lean_forward',
      'lean_back',
    ];
    for (const g of expected) {
      expect(ctrl.resolve(g)).toBe(g);
      expect(GESTURES[g]).toBeDefined();
    }
  });

  it('correctly maps gesture aliases in English and Vietnamese', () => {
    const ctrl = new GestureController();
    expect(ctrl.resolve('cheer')).toBe('praise');
    expect(ctrl.resolve('hello')).toBe('wave');
    expect(ctrl.resolve('ok')).toBe('nod');
    expect(ctrl.resolve('clapping')).toBe('clap');
    expect(ctrl.resolve('sigh')).toBe('relax');
    expect(ctrl.resolve('confused')).toBe('think');
    expect(ctrl.resolve('salute')).toBe('wave');
    expect(ctrl.resolve('laugh')).toBe('praise');
    expect(ctrl.resolve('idle')).toBe('relax');
    expect(ctrl.resolve('dance')).toBe('praise');
    expect(ctrl.resolve('wink')).toBe('wave');
    expect(ctrl.resolve('headtilt')).toBe('tilt');
    expect(ctrl.resolve('shouldershrug')).toBe('shrug');
    expect(ctrl.resolve('shrugging')).toBe('shrug');
    expect(ctrl.resolve('xinchao')).toBe('wave');
    expect(ctrl.resolve('gatdau')).toBe('nod');
    expect(ctrl.resolve('lacdau')).toBe('shake');
    expect(ctrl.resolve('chongcam')).toBe('chin_touch');
    expect(ctrl.resolve('chemiengcuoi')).toBe('mouth_cover');
    expect(ctrl.resolve('dattaylentim')).toBe('hand_on_heart');
    expect(ctrl.resolve('non-existent')).toBeNull();
  });

  it('5th-order minimum-jerk polynomial guarantees C^2 continuity', () => {
    expect(quintic(0)).toBe(0);
    expect(quintic(1)).toBe(1);
    expect(quintic(0.5)).toBe(0.5);

    // Monotonicity check
    let prev = -1;
    for (let t = 0; t <= 1.0; t += 0.05) {
      const val = quintic(t);
      expect(val).toBeGreaterThanOrEqual(prev);
      prev = val;
    }

    // Near-boundary zero 1st and 2nd derivatives
    const eps = 0.001;
    const v0 = (quintic(eps) - quintic(0)) / eps;
    expect(v0).toBeLessThan(0.001); // Velocity is zero at t=0
    const v1 = (quintic(1) - quintic(1 - eps)) / eps;
    expect(v1).toBeLessThan(0.001); // Velocity is zero at t=1
  });

  it('defines natural anatomical hand cascades', () => {
    // Natural resting hand cascade
    expect(REST_CASCADE.index).toBeCloseTo(0.20, 2);
    expect(REST_CASCADE.middle).toBeCloseTo(0.35, 2);
    expect(REST_CASCADE.ring).toBeCloseTo(0.50, 2);
    expect(REST_CASCADE.little).toBeCloseTo(0.65, 2);
    expect(REST_CASCADE.thumb).toBeCloseTo(0.15, 2);

    // Other cascades
    expect(POINT_CASCADE.index).toBeLessThan(0.05); // Extended
    expect(POINT_CASCADE.middle).toBeGreaterThan(0.8); // Curled
    expect(FIST_CASCADE.all).toBeGreaterThan(0.9);
    expect(OPEN_CASCADE.index).toBeLessThan(0.1);
  });

  it('queues and executes gestures across frames with clavicle and spine engagement', () => {
    const ctrl = new GestureController();
    expect(ctrl.isGesturing()).toBe(false);

    const queued = ctrl.playGesture('wave', 2.0);
    expect(queued).toBe(true);
    expect(ctrl.isGesturing()).toBe(true);

    const states = ctrl.update(0.1);
    expect(states.length).toBe(1);
    expect(states[0].key).toBe('wave');
    expect(states[0].w).toBeGreaterThan(0);

    const buf = new PoseBuffer();
    const hint = ctrl.apply(buf, states);
    expect(hint.happy).toBeGreaterThan(0);
    expect(buf.rot.size).toBeGreaterThan(0);

    // Verify clavicle and spine engaged
    expect(buf.rot.has('rightShoulder')).toBe(true);
    expect(buf.rot.has('spine')).toBe(true);
    expect(buf.rot.has('head')).toBe(true);
  });

  it('executes co-speech prosody gesticulation during speaking state', () => {
    const ctrl = new GestureController();
    ctrl.setSpeaking(true, 0.7);
    expect(ctrl.isSpeaking).toBe(true);
    expect(ctrl.mouthEnergy).toBe(0.7);

    const buf = new PoseBuffer();
    const states = ctrl.update(0.1, 0.7, true);
    ctrl.apply(buf, states, { mouthEnergy: 0.7, isSpeaking: true, time: 0.5 });

    // Verify co-speech beat nods and hand pulses applied
    expect(buf.rot.has('head')).toBe(true);
    expect(buf.rot.has('leftShoulder')).toBe(true);
    expect(buf.rot.has('rightLowerArm')).toBe(true);
  });

  it('clears or stops gestures when interrupted', () => {
    const ctrl = new GestureController();
    ctrl.playGesture('think');
    expect(ctrl.isGesturing()).toBe(true);
    ctrl.clear();
    expect(ctrl.isGesturing()).toBe(false);
  });
});
