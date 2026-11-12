import * as THREE from 'three';
import type { VRM } from '@pixiv/three-vrm';
import { GestureController } from './gesture-controller.js';
import { ExpressionController } from './expression-controller.js';
import { PoseBuffer, clamp, noise, eq, upperArmQuat, lowerArmQuat } from './pose-utils.js';

const _q = new THREE.Quaternion();

export class ProceduralAnimator {
  public pose = new PoseBuffer();
  public gestureController = new GestureController();
  public expressions = new ExpressionController();

  public state = 'idle';
  public stateBlend: Record<string, number> = { listening: 0, thinking: 0, speaking: 0 };

  public targetLookX = 0;
  public targetLookY = 0;
  public lookX = 0;
  public lookY = 0;
  public lastPointerTime = -100;

  public eye = { x: 0, y: 0, tx: 0, ty: 0, timer: 1 };
  public eyeGaze = { x: 0, y: 0 };
  public thinkingGaze = 0;

  private micro = { timer: 4, progress: 1, type: 0, duration: 2.2 };

  public reset(): void {
    this.pose.clearSmoothing();
    this.expressions.reset();
    this.gestureController.clear();
  }

  public setLookTarget(nx: number, ny: number, time = 0): void {
    this.targetLookX = clamp(nx, -1, 1);
    this.targetLookY = clamp(ny, -1, 1);
    this.lastPointerTime = time;
  }

  public setState(state: string): void {
    this.state = state;
    this.expressions.setSpeaking(state === 'speaking');
  }

  public setEmotion(name: string, weight = 1, hold = 5): boolean {
    return this.expressions.setEmotion(name, weight, hold);
  }

  public playGesture(name: string, duration?: number): boolean {
    return this.gestureController.playGesture(name, duration);
  }

  public isGesturing(): boolean {
    return this.gestureController.isGesturing();
  }

  public update(vrm: VRM, dt: number, time: number, { mouth = 0 }: { mouth?: number } = {}): void {
    if (!vrm?.humanoid) return;

    const buf = this.pose;
    buf.reset();

    const kState = 1 - Math.exp(-4 * dt);
    for (const key of Object.keys(this.stateBlend)) {
      const goal = this.state === key ? 1 : 0;
      this.stateBlend[key] += (goal - this.stateBlend[key]) * kState;
    }

    this.applyRest(buf);
    this.applyBreathing(buf, time);
    this.applyMicroMotion(buf, dt, time);
    this.applyState(buf, time, mouth);
    this.applyLook(buf, dt, time);

    const gestureStates = this.gestureController.update(dt);
    const hint = this.gestureController.apply(buf, gestureStates);
    this.expressions.setGestureHint(hint);

    buf.apply(vrm, dt, 14);
    this.expressions.update(vrm, dt);
  }

  private applyRest(buf: PoseBuffer): void {
    for (const side of ['left', 'right'] as const) {
      buf.set(`${side}UpperArm`, upperArmQuat(side, 0.3, 0.12, 0.05, _q));
      buf.set(`${side}LowerArm`, lowerArmQuat(side, 0.22, 0, _q));
      buf.set(`${side}Hand`, eq(0.05, 0, 0, _q));
      buf.setFingers(side, { all: 0.22, thumb: 0.15 });
      buf.get(`${side}Shoulder`);
    }
  }

  private applyBreathing(buf: PoseBuffer, t: number): void {
    const b = Math.sin(t * 1.55);
    const slow = noise(t * 0.35, 1);
    buf.add('spine', 0.012 * b, 0, 0.012 * slow);
    buf.add('chest', 0.02 * b, 0, -0.008 * slow);
    buf.add('upperChest', 0.008 * b, 0, 0);
    buf.add('leftShoulder', 0, 0, 0.012 * b);
    buf.add('rightShoulder', 0, 0, -0.012 * b);
    buf.add('neck', -0.006 * b, 0, 0);

    const shift = noise(t * 0.22, 3);
    buf.add('hips', 0, 0.02 * shift, 0.015 * shift);
    buf.offsetPos('hips', 0.006 * shift, 0.002 * b, 0);

    buf.add('leftUpperArm', 0.015 * b, 0, 0);
    buf.add('rightUpperArm', 0.015 * b, 0, 0);
    buf.add('leftLowerArm', 0, 0.02 * slow, 0);
    buf.add('rightLowerArm', 0, -0.02 * slow, 0);
  }

  private applyMicroMotion(buf: PoseBuffer, dt: number, t: number): void {
    const m = this.micro;
    m.timer -= dt;
    if (m.progress >= 1 && m.timer <= 0) {
      m.progress = 0;
      m.type = Math.floor(Math.random() * 4);
      m.timer = 5 + Math.random() * 6;
      m.duration = 1.6 + Math.random() * 1.4;
    }
    if (m.progress < 1) {
      m.progress += dt / m.duration;
      const e = Math.sin(clamp(m.progress, 0, 1) * Math.PI);
      switch (m.type) {
        case 0:
          buf.add('head', 0, 0, 0.06 * e);
          break;
        case 1:
          buf.add('head', -0.04 * e, 0.05 * e, 0);
          break;
        case 2:
          buf.add('neck', 0.03 * e, -0.06 * e, 0);
          buf.add('spine', 0, -0.02 * e, 0);
          break;
        default:
          buf.add('head', 0.04 * e, 0, -0.05 * e);
          break;
      }
    }
    buf.add('head', 0.008 * noise(t * 0.6, 5), 0.012 * noise(t * 0.4, 7), 0.01 * noise(t * 0.5, 9));
  }

  private applyState(buf: PoseBuffer, t: number, mouth: number): void {
    const { listening, thinking, speaking } = this.stateBlend;

    if (listening > 0.01) {
      const nod = Math.max(0, Math.sin(t * 0.9)) ** 8 * 0.05;
      buf.add('head', (0.02 + nod) * listening, 0, 0.07 * listening);
      buf.add('spine', 0.025 * listening, 0, 0);
    }

    if (thinking > 0.01) {
      buf.add('head', -0.06 * thinking, 0.1 * thinking, -0.06 * thinking);
      buf.add('neck', -0.02 * thinking, 0.04 * thinking, 0);
      this.thinkingGaze = thinking;
    } else {
      this.thinkingGaze = 0;
    }

    if (speaking > 0.01) {
      const beat = (mouth - 0.3) * 0.06;
      buf.add('head', beat * speaking, 0.05 * noise(t * 0.8, 11) * speaking, 0.03 * noise(t * 0.6, 13) * speaking);
      buf.add('chest', 0.012 * mouth * speaking, 0.02 * noise(t * 0.5, 15) * speaking, 0);

      const a = 0.14 * speaking;
      buf.add('rightLowerArm', 0, (0.4 + noise(t * 0.9, 17)) * a, 0);
      buf.add('leftLowerArm', 0, -(0.3 + noise(t * 0.8, 19)) * a, 0);
      buf.add('rightUpperArm', 0.04 * noise(t * 0.7, 21) * a * 4, 0, 0);
      buf.add('rightHand', 0, 0, 0.18 * noise(t * 1.3, 23) * a * 3);
    }
  }

  private applyLook(buf: PoseBuffer, dt: number, time: number): void {
    const idleFor = time - this.lastPointerTime;
    const relax = idleFor > 8 ? clamp((idleFor - 8) / 4, 0, 1) : 0;
    const goalX = this.targetLookX * (1 - relax);
    const goalY = this.targetLookY * (1 - relax);

    const k = 1 - Math.exp(-5 * dt);
    this.lookX += (goalX - this.lookX) * k;
    this.lookY += (goalY - this.lookY) * k;

    const gazeUp = (this.thinkingGaze || 0) * 0.5;
    const yaw = this.lookX * 0.5;
    const pitch = (this.lookY * 0.28 - gazeUp * 0.3) * 1;
    buf.add('neck', pitch * 0.35, yaw * 0.35, 0);
    buf.add('head', pitch * 0.65, yaw * 0.65, 0);
    buf.add('chest', 0, yaw * 0.12, 0);

    const e = this.eye;
    e.timer -= dt;
    if (e.timer <= 0) {
      e.timer = 0.7 + Math.random() * 2.2;
      e.tx = (Math.random() - 0.5) * 0.08;
      e.ty = (Math.random() - 0.5) * 0.05;
    }
    const ke = 1 - Math.exp(-18 * dt);
    e.x += (e.tx - e.x) * ke;
    e.y += (e.ty - e.y) * ke;
    this.eyeGaze = { x: this.lookX + e.x, y: this.lookY + e.y - gazeUp * 0.4 };
  }
}
