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
});
