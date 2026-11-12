import { describe, expect, it } from 'bun:test';
import { LipSyncAnimator, VISEMES } from '../src/stage/lip-sync';

describe('LipSyncAnimator', () => {
  it('exposes all 5 VRM standard vowel visemes', () => {
    expect(VISEMES).toEqual(['aa', 'ih', 'ou', 'ee', 'oh']);
  });

  it('initializes with zero mouth energy and rest state', () => {
    const animator = new LipSyncAnimator();
    expect(animator.isSpeaking).toBe(false);
    expect(animator.mouthEnergy).toBe(0);
  });

  it('updates target weights upon receiving viseme_tick from Tauri', () => {
    const animator = new LipSyncAnimator();
    const valuesSet = new Map<string, number>();
    const mockManager = {
      setValue: (name: string, val: number) => {
        valuesSet.set(name, val);
      },
    };
    const mockVrm = { expressionManager: mockManager };

    animator.onVisemeTick({
      weights: { aa: 0.8, ih: 0.2, ou: 0.0, ee: 0.1, oh: 0.0 },
      is_speaking: true,
      timestamp_ms: 100,
    });

    animator.update(mockVrm as any, 0.016);

    expect(animator.isSpeaking).toBe(true);
    expect(valuesSet.has('aa')).toBe(true);
    expect(valuesSet.get('aa')!).toBeGreaterThan(0.2);
    expect(animator.mouthEnergy).toBeGreaterThan(0.2);
  });

  it('resets all viseme values when speech stops', () => {
    const animator = new LipSyncAnimator();
    const valuesSet = new Map<string, number>();
    const mockManager = {
      setValue: (name: string, val: number) => {
        valuesSet.set(name, val);
      },
    };
    const mockVrm = { expressionManager: mockManager };

    animator.setSpeaking(true);
    animator.update(mockVrm as any, 0.1);
    expect(animator.isSpeaking).toBe(true);

    animator.setSpeaking(false);
    animator.reset();
    expect(animator.isSpeaking).toBe(false);
    expect(animator.mouthEnergy).toBe(0);
  });
});
