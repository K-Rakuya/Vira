import { describe, expect, it } from 'bun:test';
import { GestureController, GESTURES } from '../src/stage/gesture-controller';
import { PoseBuffer } from '../src/stage/pose-utils';

describe('GestureController', () => {
  it('contains expected list of procedural gestures', () => {
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
      'surprise',
      'relax',
    ];
    for (const g of expected) {
      expect(ctrl.resolve(g)).toBe(g);
    }
  });

  it('correctly maps gesture aliases', () => {
    const ctrl = new GestureController();
    expect(ctrl.resolve('cheer')).toBe('praise');
    expect(ctrl.resolve('hello')).toBe('wave');
    expect(ctrl.resolve('ok')).toBe('nod');
    expect(ctrl.resolve('clapping')).toBe('clap');
    expect(ctrl.resolve('sigh')).toBe('relax');
    expect(ctrl.resolve('confused')).toBe('think');
    expect(ctrl.resolve('shrug')).toBe('explain');
    expect(ctrl.resolve('salute')).toBe('wave');
    expect(ctrl.resolve('tilt')).toBe('shy');
    expect(ctrl.resolve('laugh')).toBe('praise');
    expect(ctrl.resolve('idle')).toBe('relax');
    expect(ctrl.resolve('dance')).toBe('praise');
    expect(ctrl.resolve('wink')).toBe('wave');
    expect(ctrl.resolve('non-existent')).toBeNull();
  });

  it('queues and executes gestures across frames', () => {
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
  });

  it('clears or stops gestures when interrupted', () => {
    const ctrl = new GestureController();
    ctrl.playGesture('think');
    expect(ctrl.isGesturing()).toBe(true);
    ctrl.clear();
    expect(ctrl.isGesturing()).toBe(false);
  });
});
