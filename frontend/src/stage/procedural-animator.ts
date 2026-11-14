import * as THREE from 'three';
import type { VRM } from '@pixiv/three-vrm';
import { GestureController } from './gesture-controller.js';
import { ExpressionController } from './expression-controller.js';
import {
  PoseBuffer,
  clamp,
  noise,
  eq,
  upperArmQuat,
  lowerArmQuat,
  GeodesicSpringDamper,
} from './pose-utils.js';

const _q = new THREE.Quaternion();
const STATE_KEYS = ['listening', 'thinking', 'speaking'] as const;
const SIDES = ['left', 'right'] as const;

/**
 * Parses both implicit kinematic descriptions (e.g. `*gật đầu*`, `*mỉm cười*`) and explicit cues
 * (`[gesture:wave]`, `[emotion:happy]`) from language model narration to coordinated avatar gestures and expressions.
 */
export function parseKinematicCues(text: string): { gesture?: string; emotion?: string } {
  const result: { gesture?: string; emotion?: string } = {};
  if (!text) return result;

  // Explicit gesture tags: [gesture:wave]
  const explicitGesture = text.match(/\[gesture:([a-zA-Z0-9_-]+)\]/i);
  if (explicitGesture) {
    result.gesture = explicitGesture[1].toLowerCase();
  }

  // Explicit emotion tags: [emotion:happy]
  const explicitEmotion = text.match(/\[emotion:([a-zA-Z0-9_-]+)\]/i);
  if (explicitEmotion) {
    result.emotion = explicitEmotion[1].toLowerCase();
  }

  const lower = text.toLowerCase();

  // Implicit gestures & emotions
  if (/\*(gật đầu|gật nhẹ|nods?)\*/i.test(lower)) {
    result.gesture = result.gesture || 'nod';
  } else if (/\*(nghiêng đầu|tilts?)\*/i.test(lower)) {
    result.gesture = result.gesture || 'tilt';
  } else if (/\*(lắc đầu|shakes?)\*/i.test(lower)) {
    result.gesture = result.gesture || 'shake';
  } else if (/\*(cúi đầu|đặt tay lên ngực|bows?)\*/i.test(lower)) {
    result.gesture = result.gesture || 'bow';
  } else if (/\*(suy nghĩ|trầm ngâm|thinks?)\*/i.test(lower)) {
    result.gesture = result.gesture || 'think';
  } else if (/\*(vẫy tay|waves?)\*/i.test(lower)) {
    result.gesture = result.gesture || 'wave';
  } else if (/\*(chỉ tay|points?)\*/i.test(lower)) {
    result.gesture = result.gesture || 'point';
  } else if (/\*(vỗ tay|claps?)\*/i.test(lower)) {
    result.gesture = result.gesture || 'clap';
  } else if (/\*(giải thích|explains?)\*/i.test(lower)) {
    result.gesture = result.gesture || 'explain';
  } else if (/\*(nhún vai|shrugs?)\*/i.test(lower)) {
    result.gesture = result.gesture || 'shrug';
  } else if (/\*(bối rối|ngại ngùng|shys?)\*/i.test(lower)) {
    result.gesture = result.gesture || 'shy';
    result.emotion = result.emotion || 'relaxed';
  } else if (/\*(hoan hô|chúc mừng|cheers?|praises?)\*/i.test(lower)) {
    result.gesture = result.gesture || 'praise';
    result.emotion = result.emotion || 'happy';
  }

  if (/\*(mỉm cười|cười tươi|cười|smiles?)\*/i.test(lower)) {
    result.emotion = result.emotion || 'happy';
  }

  return result;
}

export interface SaccadeState {
  isSaccade: boolean;
  timer: number;
  duration: number;
  startX: number;
  startY: number;
  targetX: number;
  targetY: number;
  currentX: number;
  currentY: number;
}

export class ProceduralAnimator {
  public pose = new PoseBuffer();
  public gestureController = new GestureController();
  public expressions = new ExpressionController();

  // Geodesic Spring-Dampers on SO(3)
  public headSpring = new GeodesicSpringDamper(16, 1.0); // Critically damped neck/head
  public neckSpring = new GeodesicSpringDamper(16, 1.0);
  public leftArmSpring = new GeodesicSpringDamper(12, 0.82); // Organic limb compliance
  public rightArmSpring = new GeodesicSpringDamper(12, 0.82);

  public state = 'idle';
  public stateBlend: Record<string, number> = { listening: 0, thinking: 0, speaking: 0 };

  public targetLookX = 0;
  public targetLookY = 0;
  public lookX = 0;
  public lookY = 0;
  public lastPointerTime = -100;

  public saccade: SaccadeState = {
    isSaccade: false,
    timer: 0,
    duration: 0.35,
    startX: 0,
    startY: 0,
    targetX: 0,
    targetY: 0,
    currentX: 0,
    currentY: 0,
  };
  public eyeGaze = { x: 0, y: 0 };
  public thinkingGaze = 0;

  private micro = { timer: 4, progress: 1, type: 0, duration: 2.2 };

  public reset(): void {
    this.pose.clearSmoothing();
    this.expressions.reset();
    this.gestureController.clear();
    this.headSpring.reset();
    this.neckSpring.reset();
    this.leftArmSpring.reset();
    this.rightArmSpring.reset();
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

  public update(vrm: VRM, dt: number, time: number, mouthParam: number | { mouth?: number } = 0): void {
    if (!vrm?.humanoid) return;

    const mouth = typeof mouthParam === 'number' ? mouthParam : (mouthParam?.mouth ?? 0);
    const buf = this.pose;
    buf.reset();

    const kState = 1 - Math.exp(-4 * dt);
    for (let i = 0; i < STATE_KEYS.length; i++) {
      const key = STATE_KEYS[i];
      const goal = this.state === key ? 1 : 0;
      this.stateBlend[key] += (goal - this.stateBlend[key]) * kState;
    }

    // Layer 1: Biological base layer (Rest pose, Breathing & Inverted Pendulum sway)
    this.applyRest(buf);
    this.applyBreathing(buf, time);
    this.applyMicroMotion(buf, dt, time);
    this.applyState(buf, time, mouth);
    this.applyLook(buf, dt, time);

    // Layer 2: Gesture layer blended additively without wiping out breathing
    const gestureStates = this.gestureController.update(dt);
    const hint = this.gestureController.apply(buf, gestureStates);
    this.expressions.setGestureHint(hint);

    // Layer 3: Geodesic Spring-Damper filtering on Lie group SO(3) to eliminate Gimbal Lock
    buf.set('head', this.headSpring.step(buf.get('head'), dt));
    buf.set('neck', this.neckSpring.step(buf.get('neck'), dt));
    buf.set('leftUpperArm', this.leftArmSpring.step(buf.get('leftUpperArm'), dt));
    buf.set('rightUpperArm', this.rightArmSpring.step(buf.get('rightUpperArm'), dt));

    buf.apply(vrm, dt, 14);
    this.expressions.update(vrm, dt);
  }

  private applyRest(buf: PoseBuffer): void {
    for (let i = 0; i < SIDES.length; i++) {
      const side = SIDES[i];
      buf.set(`${side}UpperArm`, upperArmQuat(side, 0.3, 0.12, 0.05, _q));
      buf.set(`${side}LowerArm`, lowerArmQuat(side, 0.22, 0, _q));
      buf.set(`${side}Hand`, eq(0.05, 0, 0, _q));
      buf.setFingers(side, { all: 0.22, thumb: 0.15 });
      buf.get(`${side}Shoulder`);
    }
  }

  private applyBreathing(buf: PoseBuffer, t: number): void {
    // Dual Lissajous curves (f1=0.08Hz, f2=0.13Hz) + Simplex noise for biological weight shift
    const tLissajous1 = t * 2 * Math.PI * 0.08;
    const tLissajous2 = t * 2 * Math.PI * 0.13;
    const swayX = 0.007 * Math.sin(tLissajous1) + 0.0035 * Math.sin(tLissajous2 + 1.2) + 0.002 * noise(t * 0.18, 31);
    const swayZ = 0.005 * Math.sin(tLissajous1 + Math.PI * 0.5) + 0.002 * Math.sin(tLissajous2 * 2);
    const hipRoll = 0.016 * Math.sin(tLissajous1) + 0.008 * noise(t * 0.22, 33);
    const hipYaw = 0.012 * Math.cos(tLissajous1 * 0.5) + 0.006 * noise(t * 0.15, 35);

    // Chest expansion breathing at 1.55 rad/s
    const b = Math.sin(t * 1.55);
    const bSlow = Math.sin(t * 0.77);
    const chestLift = 0.024 * b + 0.006 * bSlow;

    buf.add('hips', swayZ * 0.5, hipYaw, hipRoll);
    buf.offsetPos('hips', swayX, 0.003 * b, swayZ);

    buf.add('spine', 0.014 * b, 0, 0.008 * hipRoll);
    buf.add('chest', chestLift, 0, -0.006 * hipRoll);
    buf.add('upperChest', 0.012 * b, 0, 0);
    buf.add('leftShoulder', 0, 0, 0.015 * b);
    buf.add('rightShoulder', 0, 0, -0.015 * b);
    buf.add('neck', -0.008 * b, 0, 0);

    const armSlow = noise(t * 0.35, 1);
    buf.add('leftUpperArm', 0.015 * b, 0, 0);
    buf.add('rightUpperArm', 0.015 * b, 0, 0);
    buf.add('leftLowerArm', 0, 0.02 * armSlow, 0);
    buf.add('rightLowerArm', 0, -0.02 * armSlow, 0);
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

    // 1. Saccade state update
    const s = this.saccade;
    s.timer += dt;
    if (s.isSaccade) {
      const progress = clamp(s.timer / s.duration, 0, 1);
      // Minimum-jerk trajectory: 10*t^3 - 15*t^4 + 6*t^5
      const p = progress * progress * progress * (10 + progress * (-15 + 6 * progress));
      s.currentX = s.startX + (s.targetX - s.startX) * p;
      s.currentY = s.startY + (s.targetY - s.startY) * p;

      if (progress >= 1) {
        s.isSaccade = false;
        s.timer = 0;
        // Fixation interval: 250 - 450 ms
        s.duration = 0.25 + Math.random() * 0.2;
      }
    } else {
      // Fixation phase: physiological micro-tremor (~20-25 Hz)
      const tremorX = 0.003 * Math.sin(time * 140) + 0.0015 * Math.sin(time * 210);
      const tremorY = 0.002 * Math.cos(time * 160) + 0.0015 * Math.cos(time * 230);
      s.currentX = s.targetX + tremorX;
      s.currentY = s.targetY + tremorY;

      if (s.timer >= s.duration) {
        // Trigger rapid ballistic saccade: 25 - 45 ms duration (300-600 deg/sec)
        s.isSaccade = true;
        s.timer = 0;
        s.duration = 0.025 + Math.random() * 0.02; // 25-45ms
        s.startX = s.currentX;
        s.startY = s.currentY;
        s.targetX = (Math.random() - 0.5) * 0.12;
        s.targetY = (Math.random() - 0.5) * 0.08;
      }
    }

    // 2. Thinking Gaze Aversion: eyes dart upward 18° - 26° (0.31 - 0.45 rad)
    const thinkingGazeAngle = (this.thinkingGaze || 0) * (0.32 + 0.08 * Math.sin(time * 0.5));

    // 3. Vestibulo-Ocular Reflex (VOR): counter-rotate gaze against head pitch/yaw
    const vorX = -yaw * 0.35;
    const vorY = -pitch * 0.35;

    this.eyeGaze.x = this.lookX + s.currentX + vorX;
    this.eyeGaze.y = this.lookY + s.currentY - thinkingGazeAngle + vorY;
  }
}
