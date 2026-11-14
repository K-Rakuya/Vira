import type { VRM } from '@pixiv/three-vrm';
import { clamp } from './pose-utils.js';

export const EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'relaxed'] as const;
export type EmotionName = typeof EMOTIONS[number];
const EMOTION_SET = new Set<string>(EMOTIONS);

const EMOTION_GAIN: Record<string, number> = {
  happy: 0.55,
  relaxed: 0.8,
  sad: 0.7,
  angry: 0.7,
  surprised: 1.0,
};

const FALLBACKS: Record<string, Record<string, number>> = {
  surprised: { oh: 0.35 },
};

// 80ms ballistic close, 220ms gentle ease-out open with settle bounce (300ms total)
const BLINK_CLOSE_DURATION = 0.080;
const BLINK_OPEN_DURATION = 0.220;
const BLINK_TOTAL_DURATION = BLINK_CLOSE_DURATION + BLINK_OPEN_DURATION;

export class ExpressionController {
  public emotion: EmotionName = 'neutral';
  public emotionWeight = 1;
  public holdTimer = 0;
  public holdDuration = 5;
  public speaking = false;
  public thinking = false;
  public mouthEnergy = 0;
  public gestureHint: Record<string, number> = {};

  private values = new Map<string, number>();
  private targets = new Map<string, number>();
  private trackedKeys: string[] = [];
  private trackedSet = new Set<string>();

  private blinkTimer = 0;
  private nextBlink = 2.5;
  private blinkPhase = -1;
  private doubleBlink = false;

  private eyebrowFlashTimer = 0;
  private eyebrowFlashDuration = 0.22;
  private prevMouthEnergy = 0;
  private lifeTime = 0;

  public reset(): void {
    this.values.clear();
    this.targets.clear();
    this.emotion = 'neutral';
    this.gestureHint = {};
    this.blinkPhase = -1;
    this.speaking = false;
    this.thinking = false;
    this.mouthEnergy = 0;
    this.eyebrowFlashTimer = 0;
    this.prevMouthEnergy = 0;
    this.lifeTime = 0;
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
      this.mouthEnergy = 0;
      this.prevMouthEnergy = 0;
      this.eyebrowFlashTimer = 0;
    }
  }

  public setThinking(thinking: boolean): void {
    this.thinking = thinking;
  }

  public setMouthEnergy(energy: number): void {
    this.mouthEnergy = clamp(energy, 0, 1);
  }

  public setGestureHint(hint: Record<string, number>): void {
    this.gestureHint = hint || {};
  }

  public triggerBlink(forceDouble = false): void {
    this.blinkPhase = 0;
    this.blinkTimer = 0;
    this.doubleBlink = forceDouble;
  }

  private registerKey(name: string): void {
    if (!this.trackedSet.has(name)) {
      this.trackedSet.add(name);
      this.trackedKeys.push(name);
    }
  }

  private has(manager: any, name: string): boolean {
    return Boolean(manager.getExpression?.(name));
  }

  private _pushExpression(manager: any, name: string, weight: number): void {
    if (weight <= 0) return;
    if (this.has(manager, name)) {
      this.registerKey(name);
      this.targets.set(name, clamp((this.targets.get(name) || 0) + weight, 0, 1));
    } else {
      const fb = FALLBACKS[name];
      if (fb) {
        for (const k in fb) {
          this.registerKey(k);
          this.targets.set(k, clamp((this.targets.get(k) || 0) + fb[k] * weight, 0, 1));
        }
      }
    }
  }

  public update(vrm: VRM, dt: number, options?: { mouthEnergy?: number; isThinking?: boolean }): void {
    const manager = vrm?.expressionManager;
    if (!manager) return;

    this.lifeTime += dt;
    if (options?.mouthEnergy !== undefined) {
      this.mouthEnergy = clamp(options.mouthEnergy, 0, 1);
    }
    if (options?.isThinking !== undefined) {
      this.thinking = options.isThinking;
    }

    if (this.emotion !== 'neutral' && !this.speaking) {
      this.holdTimer += dt;
      if (this.holdTimer >= this.holdDuration) {
        this.emotion = 'neutral';
      }
    }

    this.targets.clear();

    // 1. Emotion Layer with Soft Asymmetric Living Dynamics
    if (this.emotion !== 'neutral') {
      const baseGain = EMOTION_GAIN[this.emotion] ?? 0.8;
      // Subtle 1.2Hz biological micro-pulsing (+/- 3%) for organic realism
      const lifeMod = 1.0 + 0.03 * Math.sin(this.lifeTime * 4.2);
      const effectiveWeight = this.emotionWeight * baseGain * lifeMod;
      this._pushExpression(manager, this.emotion, effectiveWeight);
    }

    // 2. Gesture Hint Layer
    for (const name in this.gestureHint) {
      this._pushExpression(manager, name, this.gestureHint[name]);
    }

    // 3. Brow Furrow during Thinking
    if (this.thinking || this.gestureHint.think) {
      const thinkWeight = this.thinking ? 1.0 : (this.gestureHint.think || 0);
      // Subtle brow knit: slight angry (lowers inner brow) + relaxed
      this._pushExpression(manager, 'angry', 0.18 * thinkWeight);
      this._pushExpression(manager, 'relaxed', 0.12 * thinkWeight);
    }

    // 4. Subtle Eyebrow Flashes on Vocal Emphasis
    if (this.speaking) {
      const deltaEnergy = this.mouthEnergy - this.prevMouthEnergy;
      // Acoustic onset trigger: rapid energy spike or sustained high vocal energy
      if ((deltaEnergy > 0.18 || this.mouthEnergy > 0.6) && this.eyebrowFlashTimer <= 0) {
        this.eyebrowFlashTimer = this.eyebrowFlashDuration;
      }
      this.prevMouthEnergy = this.mouthEnergy;

      if (this.eyebrowFlashTimer > 0) {
        this.eyebrowFlashTimer -= dt;
        const progress = clamp(this.eyebrowFlashTimer / this.eyebrowFlashDuration, 0, 1);
        const flashCurve = Math.sin(progress * Math.PI);
        // Eyebrow flash via surprised or lookup
        this._pushExpression(manager, 'surprised', flashCurve * 0.22);
      }
    } else {
      this.prevMouthEnergy = 0;
      this.eyebrowFlashTimer = 0;
    }

    // 5. Zero-Allocation Exponential Smoothing on Expression Channels
    const k = 1 - Math.exp(-10 * dt);
    const keyCount = this.trackedKeys.length;
    for (let i = 0; i < keyCount; i++) {
      const name = this.trackedKeys[i];
      const cur = this.values.get(name) || 0;
      const targetVal = this.targets.get(name) || 0;
      const next = cur + (targetVal - cur) * k;

      if (next < 0.002 && targetVal === 0) {
        if (cur !== 0) {
          this.values.set(name, 0);
          manager.setValue(name, 0);
        }
      } else {
        this.values.set(name, next);
        manager.setValue(name, next);
      }
    }

    // 6. Natural Biomechanical Eyelid Blink Dynamics
    this.updateBlink(manager, dt);
  }

  private updateBlink(manager: any, dt: number): void {
    const hasBlink = this.has(manager, 'blink');
    const hasBlinkL = this.has(manager, 'blinkLeft') || this.has(manager, 'blink_l');
    const hasBlinkR = this.has(manager, 'blinkRight') || this.has(manager, 'blink_r');
    if (!hasBlink && !hasBlinkL && !hasBlinkR) return;

    this.blinkTimer += dt;
    if (this.blinkPhase < 0 && this.blinkTimer >= this.nextBlink) {
      this.blinkPhase = 0;
      this.blinkTimer = 0;
      this.doubleBlink = Math.random() < 0.14;
      this.nextBlink = 2.4 + Math.random() * 3.6;
    }

    if (this.blinkPhase >= 0) {
      this.blinkPhase += dt;
      let w: number;

      if (this.blinkPhase < BLINK_CLOSE_DURATION) {
        // Fast 80ms ballistic closing: accelerating eyelid descent
        const tau = this.blinkPhase / BLINK_CLOSE_DURATION;
        w = Math.sin(tau * Math.PI * 0.5);
      } else if (this.blinkPhase < BLINK_TOTAL_DURATION) {
        // Gentle 220ms ease-out opening with physiological settle bounce
        const tau = (this.blinkPhase - BLINK_CLOSE_DURATION) / BLINK_OPEN_DURATION;
        // Ease-out quadratic decay
        const easeOut = (1 - tau) * (1 - tau);
        // Subtle levator settle bounce (~3% overshoot rebound around 60-75% opening)
        const bounce = 0.08 * Math.sin(tau * Math.PI) * (1 - tau);
        w = clamp(easeOut - bounce, 0, 1);
      } else {
        w = 0;
        if (this.doubleBlink) {
          this.doubleBlink = false;
          this.blinkPhase = 0;
        } else {
          this.blinkPhase = -1;
        }
      }

      // Suppress blink slightly if squinting happily or relaxed
      const squint = Math.max(this.values.get('happy') || 0, this.values.get('relaxed') || 0);
      const blinkWeight = clamp(w * (1 - squint * 0.62), 0, 1);

      if (hasBlink) {
        manager.setValue('blink', blinkWeight);
      }
      // Natural asymmetric eyelid response across left & right hemifaces
      if (hasBlinkL) {
        const leftName = this.has(manager, 'blinkLeft') ? 'blinkLeft' : 'blink_l';
        manager.setValue(leftName, clamp(blinkWeight * 1.01, 0, 1));
      }
      if (hasBlinkR) {
        const rightName = this.has(manager, 'blinkRight') ? 'blinkRight' : 'blink_r';
        manager.setValue(rightName, clamp(blinkWeight * 0.99, 0, 1));
      }
    } else {
      if (hasBlink) manager.setValue('blink', 0);
      if (hasBlinkL) {
        const leftName = this.has(manager, 'blinkLeft') ? 'blinkLeft' : 'blink_l';
        manager.setValue(leftName, 0);
      }
      if (hasBlinkR) {
        const rightName = this.has(manager, 'blinkRight') ? 'blinkRight' : 'blink_r';
        manager.setValue(rightName, 0);
      }
    }
  }
}
