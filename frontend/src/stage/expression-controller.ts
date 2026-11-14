import type { VRM } from '@pixiv/three-vrm';
import { clamp } from './pose-utils.js';

export const EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'relaxed'] as const;
export type EmotionName = typeof EMOTIONS[number];
const EMOTION_SET = new Set<string>(EMOTIONS);

const EMOTION_GAIN: Record<string, number> = {
  happy: 0.5,
  relaxed: 0.8,
  sad: 0.7,
  angry: 0.7,
  surprised: 1.0,
};

const FALLBACKS: Record<string, Record<string, number>> = {
  surprised: { oh: 0.35 },
};

export class ExpressionController {
  public emotion: EmotionName = 'neutral';
  public emotionWeight = 1;
  public holdTimer = 0;
  public holdDuration = 5;
  public speaking = false;
  public gestureHint: Record<string, number> = {};

  private values = new Map<string, number>();
  private targets = new Map<string, number>();
  private activeNames = new Set<string>();
  private blinkTimer = 0;
  private nextBlink = 2.5;
  private blinkPhase = -1;
  private doubleBlink = false;

  public reset(): void {
    this.values.clear();
    this.emotion = 'neutral';
    this.gestureHint = {};
    this.blinkPhase = -1;
  }

  public setEmotion(name: string, weight = 1, hold = 5): boolean {
    const key = String(name || 'neutral').toLowerCase();
    if (!EMOTION_SET.has(key)) return false;
    this.emotion = key as EmotionName;
    this.emotionWeight = clamp(weight, 0, 1);
    this.holdDuration = hold;
    this.holdTimer = 0;
    return true;
  }

  public setSpeaking(speaking: boolean): void {
    this.speaking = speaking;
    if (!speaking) {
      this.holdTimer = Math.max(0, this.holdTimer - this.holdDuration + 1.5);
    }
  }

  public setGestureHint(hint: Record<string, number>): void {
    this.gestureHint = hint || {};
  }

  private has(manager: any, name: string): boolean {
    return Boolean(manager.getExpression?.(name));
  }

  public update(vrm: VRM, dt: number): void {
    const manager = vrm?.expressionManager;
    if (!manager) return;

    if (this.emotion !== 'neutral' && !this.speaking) {
      this.holdTimer += dt;
      if (this.holdTimer >= this.holdDuration) {
        this.emotion = 'neutral';
      }
    }

    this.targets.clear();
    const push = (name: string, weight: number) => {
      if (weight <= 0) return;
      const resolved = this.has(manager, name) ? { [name]: weight } : FALLBACKS[name] || {};
      for (const [k, v] of Object.entries(resolved)) {
        this.targets.set(k, clamp((this.targets.get(k) || 0) + v, 0, 1));
      }
    };

    if (this.emotion !== 'neutral') {
      push(this.emotion, this.emotionWeight * (EMOTION_GAIN[this.emotion] ?? 0.8));
    }
    for (const [name, w] of Object.entries(this.gestureHint)) {
      push(name, w);
    }

    const k = 1 - Math.exp(-9 * dt);
    this.activeNames.clear();
    for (const k of this.values.keys()) this.activeNames.add(k);
    for (const k of this.targets.keys()) this.activeNames.add(k);

    for (const name of this.activeNames) {
      const cur = this.values.get(name) || 0;
      const targetVal = this.targets.get(name) || 0;
      const next = cur + (targetVal - cur) * k;
      if (next < 0.002 && !this.targets.has(name)) {
        this.values.delete(name);
        manager.setValue(name, 0);
      } else {
        this.values.set(name, next);
        manager.setValue(name, next);
      }
    }

    this.updateBlink(manager, dt);
  }

  private updateBlink(manager: any, dt: number): void {
    if (!this.has(manager, 'blink')) return;

    this.blinkTimer += dt;
    if (this.blinkPhase < 0 && this.blinkTimer >= this.nextBlink) {
      this.blinkPhase = 0;
      this.blinkTimer = 0;
      this.doubleBlink = Math.random() < 0.15;
      this.nextBlink = 2.2 + Math.random() * 3.8;
    }

    if (this.blinkPhase >= 0) {
      this.blinkPhase += dt;
      const close = 0.07;
      const open = 0.14;
      let w: number;
      if (this.blinkPhase < close) {
        w = this.blinkPhase / close;
      } else if (this.blinkPhase < close + open) {
        w = 1 - (this.blinkPhase - close) / open;
      } else {
        w = 0;
        if (this.doubleBlink) {
          this.doubleBlink = false;
          this.blinkPhase = 0;
        } else {
          this.blinkPhase = -1;
        }
      }

      const squint = Math.max(this.values.get('happy') || 0, this.values.get('relaxed') || 0);
      manager.setValue('blink', clamp(w * (1 - squint * 0.6), 0, 1));
    } else {
      manager.setValue('blink', 0);
    }
  }
}
