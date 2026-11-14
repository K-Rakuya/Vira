import { describe, expect, it } from 'bun:test';
import { ExpressionController } from '../src/stage/expression-controller';

describe('ExpressionController', () => {
  it('initializes with neutral emotion', () => {
    const ctrl = new ExpressionController();
    expect(ctrl.emotion).toBe('neutral');
    expect(ctrl.speaking).toBe(false);
  });

  it('validates and accepts standard emotions', () => {
    const ctrl = new ExpressionController();
    expect(ctrl.setEmotion('happy', 0.8, 3)).toBe(true);
    expect(ctrl.emotion).toBe('happy');
    expect(ctrl.emotionWeight).toBe(0.8);
    expect(ctrl.holdDuration).toBe(3);

    expect(ctrl.setEmotion('surprised')).toBe(true);
    expect(ctrl.emotion).toBe('surprised');

    // Invalid emotion rejected
    expect(ctrl.setEmotion('supercalifragilistic')).toBe(false);
    expect(ctrl.emotion).toBe('surprised');
  });

  it('handles speaking state transitions', () => {
    const ctrl = new ExpressionController();
    ctrl.setEmotion('happy', 1, 5);
    ctrl.setSpeaking(true);
    expect(ctrl.speaking).toBe(true);

    ctrl.setSpeaking(false);
    expect(ctrl.speaking).toBe(false);
  });

  it('updates expressions on VRM expression manager', () => {
    const valuesSet = new Map<string, number>();
    const mockManager = {
      getExpression: (name: string) => ({ name }),
      setValue: (name: string, val: number) => {
        valuesSet.set(name, val);
      },
    };

    const mockVrm = {
      expressionManager: mockManager,
    };

    const ctrl = new ExpressionController();
    ctrl.setEmotion('happy', 1.0);
    ctrl.update(mockVrm as any, 0.1);

    expect(valuesSet.has('happy')).toBe(true);
    expect(valuesSet.get('happy')!).toBeGreaterThan(0);
  });

  it('executes natural blink dynamics with 80ms close and 220ms open', () => {
    const valuesSet = new Map<string, number>();
    const mockManager = {
      getExpression: (name: string) => ({ name }),
      setValue: (name: string, val: number) => {
        valuesSet.set(name, val);
      },
    };
    const mockVrm = { expressionManager: mockManager };

    const ctrl = new ExpressionController();
    ctrl.triggerBlink(false);

    // Mid closing phase (40ms into 80ms close)
    ctrl.update(mockVrm as any, 0.04);
    const midClose = valuesSet.get('blink') ?? 0;
    expect(midClose).toBeGreaterThan(0.5);

    // Apex at end of close (80ms)
    ctrl.update(mockVrm as any, 0.04);
    const apex = valuesSet.get('blink') ?? 0;
    expect(apex).toBeCloseTo(1.0, 1);

    // Opening phase (100ms into opening phase)
    ctrl.update(mockVrm as any, 0.10);
    const opening = valuesSet.get('blink') ?? 0;
    expect(opening).toBeLessThan(apex);
    expect(opening).toBeGreaterThan(0);

    // Complete blink (after 320ms total)
    ctrl.update(mockVrm as any, 0.15);
    expect(valuesSet.get('blink') ?? 0).toBe(0);
  });

  it('furrows brows during thinking state', () => {
    const valuesSet = new Map<string, number>();
    const mockManager = {
      getExpression: (name: string) => ({ name }),
      setValue: (name: string, val: number) => {
        valuesSet.set(name, val);
      },
    };
    const mockVrm = { expressionManager: mockManager };

    const ctrl = new ExpressionController();
    ctrl.setThinking(true);
    ctrl.update(mockVrm as any, 0.1);

    expect(valuesSet.has('angry')).toBe(true);
    expect(valuesSet.get('angry')!).toBeGreaterThan(0);
    expect(valuesSet.has('relaxed')).toBe(true);
    expect(valuesSet.get('relaxed')!).toBeGreaterThan(0);
  });

  it('triggers subtle eyebrow flashes on vocal emphasis', () => {
    const valuesSet = new Map<string, number>();
    const mockManager = {
      getExpression: (name: string) => ({ name }),
      setValue: (name: string, val: number) => {
        valuesSet.set(name, val);
      },
    };
    const mockVrm = { expressionManager: mockManager };

    const ctrl = new ExpressionController();
    ctrl.setSpeaking(true);
    ctrl.setMouthEnergy(0.8);
    ctrl.update(mockVrm as any, 0.05);

    expect(valuesSet.has('surprised')).toBe(true);
    expect(valuesSet.get('surprised')!).toBeGreaterThan(0);
  });
});
