import type { VRM } from '@pixiv/three-vrm';
import { clamp } from './pose-utils.js';

export const VISEMES = ['aa', 'ih', 'ou', 'ee', 'oh'] as const;
export type VisemeName = typeof VISEMES[number];

export interface VisemeWeightsMap {
  aa: number;
  ih: number;
  ou: number;
  ee: number;
  oh: number;
}

export interface VisemeTickEvent {
  weights: VisemeWeightsMap;
  sample_clock?: number;
  timestamp_ms?: number;
  is_speaking?: boolean;
}

export class LipSyncAnimator {
  public isSpeaking = false;
  private currentWeights: Record<VisemeName, number> = {
    aa: 0,
    ih: 0,
    ou: 0,
    ee: 0,
    oh: 0,
  };
  private targetWeights: Record<VisemeName, number> = {
    aa: 0,
    ih: 0,
    ou: 0,
    ee: 0,
    oh: 0,
  };
  private lastTickTime = 0;
  private envelope = 0;

  // Fallback simulation parameters (when IPC is offline or during testing)
  private simPhase = 0;
  private simIndex = 0;
  private simLastCycle = 0;

  public setSpeaking(speaking: boolean): void {
    this.isSpeaking = speaking;
    if (!speaking) {
      for (const v of VISEMES) {
        this.targetWeights[v] = 0;
      }
      this.envelope = 0;
    }
  }

  public get mouthEnergy(): number {
    return this.envelope;
  }

  /**
   * Called whenever a `viseme_tick` event is received from Tauri IPC (60Hz)
   */
  public onVisemeTick(payload: VisemeTickEvent): void {
    this.lastTickTime = performance.now();
    if (typeof payload.is_speaking === 'boolean') {
      this.isSpeaking = payload.is_speaking;
    }

    if (!payload.weights) return;

    for (const v of VISEMES) {
      this.targetWeights[v] = clamp(payload.weights[v] ?? 0, 0, 1);
    }
  }

  /**
   * Per-frame update called in Three.js render loop
   */
  public update(vrm: VRM, dt: number): void {
    const manager = vrm?.expressionManager;
    if (!manager) return;

    const now = performance.now();
    const hasRecentIpcTick = (now - this.lastTickTime) < 300;

    if (hasRecentIpcTick) {
      // Direct high-fidelity IPC driven visemes with light smoothing
      const k = 1 - Math.exp(-32 * dt);
      let maxWeight = 0;

      for (const v of VISEMES) {
        const target = this.targetWeights[v];
        this.currentWeights[v] += (target - this.currentWeights[v]) * k;
        const val = this.currentWeights[v] < 0.003 ? 0 : this.currentWeights[v];
        manager.setValue(v, val);
        if (val > maxWeight) maxWeight = val;
      }
      this.envelope = maxWeight;
    } else if (this.isSpeaking) {
      // Procedural synthetic lip-sync fallback if speaking but no raw audio ticks
      this.simPhase += dt * (7.5 + Math.sin(this.simPhase * 0.31) * 1.5);
      const syllable = Math.max(0, Math.sin(this.simPhase)) ** 0.8;
      const phrase = 0.65 + 0.35 * Math.sin(this.simPhase * 0.23 + 1.3);
      const target = clamp(syllable * phrase * 0.75, 0, 0.85);

      const cycle = Math.floor(this.simPhase / Math.PI);
      if (cycle !== this.simLastCycle) {
        this.simLastCycle = cycle;
        this.simIndex = (this.simIndex + 1 + Math.floor(Math.random() * 2)) % VISEMES.length;
      }

      const follow = 1 - Math.exp((target > this.envelope ? 26 : 16) * -dt);
      this.envelope += (target - this.envelope) * follow;

      const main = VISEMES[this.simIndex];
      const side = VISEMES[(this.simIndex + 2) % VISEMES.length];
      const k = 1 - Math.exp(-20 * dt);

      for (const v of VISEMES) {
        const goal = v === main ? this.envelope : v === side ? this.envelope * 0.25 : 0;
        this.currentWeights[v] += (goal - this.currentWeights[v]) * k;
        manager.setValue(v, this.currentWeights[v] < 0.003 ? 0 : this.currentWeights[v]);
      }
    } else {
      // Fade out mouth to rest
      const k = 1 - Math.exp(-20 * dt);
      this.envelope *= (1 - k);
      for (const v of VISEMES) {
        if (this.currentWeights[v] > 0.001) {
          this.currentWeights[v] *= (1 - k);
          manager.setValue(v, this.currentWeights[v] < 0.002 ? 0 : this.currentWeights[v]);
        }
      }
    }
  }

  public reset(): void {
    this.isSpeaking = false;
    this.envelope = 0;
    for (const v of VISEMES) {
      this.currentWeights[v] = 0;
      this.targetWeights[v] = 0;
    }
  }
}
