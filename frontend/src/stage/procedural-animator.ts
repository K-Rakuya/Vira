import * as THREE from 'three';
import type { VRM } from '@pixiv/three-vrm';
import { GestureController } from './gesture-controller.js';
import { ExpressionController } from './expression-controller.js';
import {
  PoseBuffer,
  clamp,
  lerp,
  noise,
  eq,
  upperArmQuat,
  lowerArmQuat,
  GeodesicSpringDamper,
  solveTwoBoneArmIK,
  getSpatialAnchor,
  applyScapulohumeralRhythm,
  type Side,
  type FingerCurls,
} from './pose-utils.js';

const _q = new THREE.Quaternion();
const STATE_KEYS = ['listening', 'thinking', 'speaking'] as const;
const SIDES = ['left', 'right'] as const;

export interface KinematicCueResult {
  gesture?: string;
  emotion?: string;
  action?: string;
  lateral?: 'left' | 'right' | 'both';
  lean?: 'forward' | 'back';
}

/**
 * Samples an exponential distribution to generate spontaneous Poisson-distributed blink intervals.
 * Mean: 3.8s (bounded strictly between 2.2s and 6.0s).
 */
function samplePoissonBlink(meanInterval = 3.8): number {
  const u = Math.max(1e-4, Math.random());
  const interval = -meanInterval * Math.log(u);
  return clamp(interval, 2.2, 6.0);
}

/**
 * Parses both implicit kinematic descriptions (Vietnamese & English natural language action descriptions)
 * and explicit cues (`[gesture:wave]`, `[emotion:happy]`, `[action:chin]`) to coordinate avatar gestures,
 * parametric IK actions, postures, and expressions.
 */
export function parseKinematicCues(text: string): KinematicCueResult {
  const result: KinematicCueResult = {};
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

  // Explicit action tags: [action:chin]
  const explicitAction = text.match(/\[action:([a-zA-Z0-9_-]+)\]/i);
  if (explicitAction) {
    result.action = explicitAction[1].toLowerCase();
  }

  const lower = text.toLowerCase();

  // Extract all delimited action phrases (*...*, (...), [...], _..._)
  const phrases: string[] = [];
  const matches = text.matchAll(/[\*\(\[_]([^\*\)\]_]+)[\*\)\]_]/g);
  for (const m of matches) {
    phrases.push(m[1].toLowerCase());
  }
  phrases.push(lower);

  for (const phrase of phrases) {
    // 1. Spatial Anatomical Actions (Generative Parametric IK)
    if (/chống cằm|chống tay lên cằm|resting chin|chin on hand/i.test(phrase)) {
      result.action = result.action || 'chin';
      result.lateral = result.lateral || 'right';
      result.gesture = result.gesture || 'think';
      result.emotion = result.emotion || 'thinking';
    } else if (/áp tay lên má|ôm má|chạm má|resting cheek|hand on cheek/i.test(phrase)) {
      result.action = result.action || 'cheek';
      result.lateral = result.lateral || 'right';
      result.gesture = result.gesture || 'shy';
      result.emotion = result.emotion || 'relaxed';
    } else if (/che miệng|che miệng cười|bịt miệng|covering mouth|hand over mouth/i.test(phrase)) {
      result.action = result.action || 'mouth';
      result.lateral = result.lateral || 'right';
      result.gesture = result.gesture || 'shy';
      result.emotion = result.emotion || 'happy';
    } else if (/đặt tay lên tim|đặt tay lên ngực|hand on heart|hand on chest/i.test(phrase)) {
      result.action = result.action || 'chestHeart';
      result.lateral = result.lateral || 'right';
      result.gesture = result.gesture || 'bow';
      result.emotion = result.emotion || 'happy';
    } else if (/chắp tay|chắp tay trước ngực|cầu nguyện|clasped hands|praying hands/i.test(phrase)) {
      result.action = result.action || 'chestCenter';
      result.lateral = result.lateral || 'both';
      result.gesture = result.gesture || 'bow';
      result.emotion = result.emotion || 'relaxed';
    } else if (/day thái dương|xoa thái dương|chỉ trán|xoa trán|sờ trán|rubbing temples?|touching forehead/i.test(phrase)) {
      result.action = result.action || 'temple';
      result.lateral = result.lateral || 'right';
      result.gesture = result.gesture || 'think';
      result.emotion = result.emotion || 'thinking';
    } else if (/đưa hai tay ra trước muốn ôm|dang tay ôm|muốn ôm|dang rộng tay|open embrace|reaching out to hug/i.test(phrase)) {
      result.action = result.action || 'openEmbrace';
      result.lateral = result.lateral || 'both';
      result.gesture = result.gesture || 'praise';
      result.emotion = result.emotion || 'happy';
    } else if (/gõ nhịp|xoa hai bàn tay|xoa tay|đặt tay lên bàn|tapping fingers|rubbing hands/i.test(phrase)) {
      result.action = result.action || 'lap';
      result.lateral = result.lateral || 'both';
      result.gesture = result.gesture || 'explain';
      result.emotion = result.emotion || 'relaxed';
    }

    // 2. Torso Posture & Lean
    if (/cúi lại gần|tiến lại gần|nghiêng người về phía trước|cúi người|leaning forward|lean in/i.test(phrase)) {
      result.lean = result.lean || 'forward';
    } else if (/ngả người ra sau|nghiêng người ra sau|lùi lại|leaning back|lean back/i.test(phrase)) {
      result.lean = result.lean || 'back';
    }

    // 3. Canonical Implicit Gestures
    if (/gật đầu|gật nhẹ|nods?/i.test(phrase)) {
      result.gesture = result.gesture || 'nod';
    } else if (/nghiêng đầu|tilts?/i.test(phrase)) {
      result.gesture = result.gesture || 'tilt';
    } else if (/lắc đầu|shakes?/i.test(phrase)) {
      result.gesture = result.gesture || 'shake';
    } else if (/cúi đầu|bows?/i.test(phrase)) {
      result.gesture = result.gesture || 'bow';
    } else if (/suy nghĩ|trầm ngâm|thinks?/i.test(phrase)) {
      result.gesture = result.gesture || 'think';
    } else if (/vẫy tay|waves?/i.test(phrase)) {
      result.gesture = result.gesture || 'wave';
    } else if (/chỉ tay|points?/i.test(phrase)) {
      result.gesture = result.gesture || 'point';
    } else if (/vỗ tay|claps?/i.test(phrase)) {
      result.gesture = result.gesture || 'clap';
    } else if (/giải thích|explains?/i.test(phrase)) {
      result.gesture = result.gesture || 'explain';
    } else if (/nhún vai|shrugs?/i.test(phrase)) {
      result.gesture = result.gesture || 'shrug';
    } else if (/bối rối|ngại ngùng|shys?/i.test(phrase)) {
      result.gesture = result.gesture || 'shy';
      result.emotion = result.emotion || 'relaxed';
    } else if (/hoan hô|chúc mừng|cheers?|praises?/i.test(phrase)) {
      result.gesture = result.gesture || 'praise';
      result.emotion = result.emotion || 'happy';
    }

    // 4. Emotions
    if (/mỉm cười|cười tươi|cười|smiles?/i.test(phrase)) {
      result.emotion = result.emotion || 'happy';
    }
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

  // Coordinated Full-Body Kinematic Chain Nodes
  public chainHead = { yaw: 0, pitch: 0 };
  public chainNeck = { yaw: 0, pitch: 0 };
  public chainUpperChest = { yaw: 0, pitch: 0 };
  public chainChest = { yaw: 0, pitch: 0 };
  public chainSpine = { yaw: 0, pitch: 0 };

  private prevHeadYaw = 0;
  private prevHeadPitch = 0;

  // Contrapposto & Spontaneous Weight Shifting State
  public contrapposto = {
    timer: 15.0,
    dominanceTarget: 1.0, // +1 = right leg dominant, -1 = left leg dominant, 0 = neutral
    dominanceCurrent: 0.0,
    startDominance: 0.0,
    progress: 1.0,
    duration: 3.0,
  };

  // Saccade & Ocular-Motor State
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

  // Saccade-Coupled & Poisson Blinking State
  public blinkTimer = 0;
  public nextBlinkInterval = 3.5;

  // Generative Parametric IK Action State
  public generativeAction = {
    active: false,
    action: '',
    timer: 0,
    duration: 3.2,
    lateral: 'right' as 'left' | 'right' | 'both',
    lean: 0, // forward (+1) or back (-1)
    leftTarget: new THREE.Vector3(),
    leftPole: new THREE.Vector3(),
    rightTarget: new THREE.Vector3(),
    rightPole: new THREE.Vector3(),
    leftFingers: undefined as FingerCurls | undefined,
    rightFingers: undefined as FingerCurls | undefined,
    outUpperL: new THREE.Quaternion(),
    outLowerL: new THREE.Quaternion(),
    outUpperR: new THREE.Quaternion(),
    outLowerR: new THREE.Quaternion(),
  };

  private micro = { timer: 4, progress: 1, type: 0, duration: 2.2 };

  public reset(): void {
    this.pose.clearSmoothing();
    this.expressions.reset();
    this.gestureController.clear();
    this.headSpring.reset();
    this.neckSpring.reset();
    this.leftArmSpring.reset();
    this.rightArmSpring.reset();
    this.generativeAction.active = false;
    this.chainHead.yaw = 0;
    this.chainHead.pitch = 0;
    this.chainNeck.yaw = 0;
    this.chainNeck.pitch = 0;
    this.chainUpperChest.yaw = 0;
    this.chainUpperChest.pitch = 0;
    this.chainChest.yaw = 0;
    this.chainChest.pitch = 0;
    this.chainSpine.yaw = 0;
    this.chainSpine.pitch = 0;
    this.prevHeadYaw = 0;
    this.prevHeadPitch = 0;
    this.contrapposto.dominanceCurrent = 0;
    this.contrapposto.dominanceTarget = 1.0;
    this.contrapposto.progress = 1.0;
    this.contrapposto.timer = 15.0;
    this.blinkTimer = 0;
    this.nextBlinkInterval = samplePoissonBlink(3.8);
  }

  public setLookTarget(nx: number, ny: number, time = 0): void {
    const clampedX = clamp(nx, -1, 1);
    const clampedY = clamp(ny, -1, 1);
    const deltaTarget = Math.hypot(clampedX - this.targetLookX, clampedY - this.targetLookY);
    // Large saccade-coupled blink trigger on sudden gaze shift (>20 degrees)
    if (deltaTarget > 0.5 && this.blinkTimer > 1.2) {
      this.triggerBlink();
    }
    this.targetLookX = clampedX;
    this.targetLookY = clampedY;
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
    return this.gestureController.isGesturing() || this.generativeAction.active;
  }

  /**
   * Triggers a natural blink. Dispatches to ExpressionController if available.
   */
  public triggerBlink(forceDouble = false): void {
    this.blinkTimer = 0;
    this.nextBlinkInterval = samplePoissonBlink(3.8);
    if (typeof (this.expressions as any).triggerBlink === 'function') {
      (this.expressions as any).triggerBlink(forceDouble);
    }
  }

  /**
   * Synthesizes and executes a generative parametric IK action from an anatomical spatial anchor.
   */
  public playAction(
    actionName: string,
    lateral: 'left' | 'right' | 'both' = 'right',
    duration = 3.2,
    lean?: 'forward' | 'back'
  ): boolean {
    const act = this.generativeAction;
    act.action = actionName;
    act.lateral = lateral;
    act.duration = duration;
    act.timer = 0;
    act.lean = lean === 'forward' ? 1 : lean === 'back' ? -1 : 0;

    let found = false;
    if (lateral === 'left' || lateral === 'both') {
      const f = getSpatialAnchor(actionName, 'left', act.leftTarget, act.leftPole);
      if (f !== undefined) {
        act.leftFingers = f;
        found = true;
      }
    }
    if (lateral === 'right' || lateral === 'both') {
      const f = getSpatialAnchor(actionName, 'right', act.rightTarget, act.rightPole);
      if (f !== undefined) {
        act.rightFingers = f;
        found = true;
      }
    }

    if (found) {
      act.active = true;
      return true;
    }
    return false;
  }

  /**
   * Applies kinematic cues parsed from dialogue or action descriptions.
   */
  public applyKinematicCues(cues: KinematicCueResult): void {
    if (cues.action) {
      this.playAction(cues.action, cues.lateral ?? 'right', 3.2, cues.lean);
    } else if (cues.gesture) {
      this.playGesture(cues.gesture);
    }

    if (cues.emotion) {
      this.setEmotion(cues.emotion);
    }
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

    // Layer 1: Biological base layer (Rest pose, Contrapposto weight shift, Asymmetric breathing & Kinematic chain)
    this.applyRest(buf);
    this.applyContrapposto(buf, dt, time);
    this.applyBreathing(buf, time);
    this.applyMicroMotion(buf, dt, time);
    this.applyState(buf, time, mouth);
    this.applyLook(buf, dt, time);

    // Layer 2A: Generative Parametric IK Action Layer
    this.applyGenerativeAction(buf, dt);

    // Layer 2B: Gesture layer blended additively without wiping out breathing
    const gestureStates = this.gestureController.update(dt);
    const hint = this.gestureController.apply(buf, gestureStates);
    this.expressions.setGestureHint(hint);

    // Layer 3: Automatic Scapulohumeral Rhythm Coupling (1 deg clavicle per 3 deg upper arm elevation)
    applyScapulohumeralRhythm(buf);

    // Layer 4: Geodesic Spring-Damper filtering on Lie group SO(3) to eliminate Gimbal Lock
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

  /**
   * Spontaneous biological shifting between left and right leg dominance (12-25s intervals).
   * Rolls pelvis by 2-3 degrees, offsets hip laterally, and counter-tilts the spine and shoulders
   * in opposite direction (contrapposto principle) so head and eyes remain horizontally stable.
   */
  private applyContrapposto(buf: PoseBuffer, dt: number, t: number): void {
    const c = this.contrapposto;
    c.timer -= dt;

    if (c.timer <= 0) {
      if (c.dominanceTarget > 0.5) {
        c.dominanceTarget = Math.random() < 0.75 ? -1.0 : 0.0;
      } else if (c.dominanceTarget < -0.5) {
        c.dominanceTarget = Math.random() < 0.75 ? 1.0 : 0.0;
      } else {
        c.dominanceTarget = Math.random() < 0.5 ? 1.0 : -1.0;
      }
      c.startDominance = c.dominanceCurrent;
      c.progress = 0;
      c.duration = 2.6 + Math.random() * 0.8;
      c.timer = 12.0 + Math.random() * 13.0; // Spontaneous 12-25s interval
    }

    if (c.progress < 1.0) {
      c.progress = Math.min(1.0, c.progress + dt / c.duration);
      const p = c.progress;
      // 5th-order minimum-jerk polynomial for C2 continuous weight transfer
      const s5 = p * p * p * (10 + p * (-15 + 6 * p));
      c.dominanceCurrent = lerp(c.startDominance, c.dominanceTarget, s5);
    }

    const w = c.dominanceCurrent; // [-1.0, 1.0]

    // 1. Pelvis Roll: ~2.4 degrees (0.042 rad) on engaged weight-bearing side
    const pelvisRoll = -0.042 * w;
    // 2. Lateral Pelvis Shift: ~2.6 cm
    const pelvisX = -0.026 * w;
    // 3. Sagittal weight transfer dip: subtle vertical drop as weight shifts between legs
    const hipDip = -0.005 * (1.0 - w * w);
    // 4. Pelvis Yaw: relaxed free hip swings slightly forward
    const pelvisYaw = 0.016 * w;

    // Inverted pendulum Lissajous sway blended on top of postural stance
    const tLissajous1 = t * 2 * Math.PI * 0.08;
    const tLissajous2 = t * 2 * Math.PI * 0.13;
    const swayX = 0.006 * Math.sin(tLissajous1) + 0.003 * Math.sin(tLissajous2 + 1.2) + 0.002 * noise(t * 0.18, 31);
    const swayZ = 0.005 * Math.sin(tLissajous1 + Math.PI * 0.5) + 0.002 * Math.sin(tLissajous2 * 2);

    buf.add('hips', swayZ * 0.5, pelvisYaw, pelvisRoll);
    buf.offsetPos('hips', pelvisX + swayX, hipDip, swayZ);

    // 5. Contrapposto Counter-Tilt:
    // Lumbar spine counter-curves to balance weight over supporting foot
    buf.add('spine', 0, -0.008 * w, 0.026 * w);
    // Thoracic chest continues S-curve
    buf.add('chest', 0, 0, 0.016 * w);
    // Shoulder girdle acromial tilt is in OPPOSITE direction of pelvic axis
    buf.add('leftShoulder', 0, 0, 0.025 * w);
    buf.add('rightShoulder', 0, 0, 0.025 * w);
    // Net roll (Pelvis -0.042 + Spine +0.026 + Chest +0.016 = 0.000 rad)
    // Head and eyes remain horizontally stable!
  }

  /**
   * Asymmetric two-phase biological respiration model:
   * - Active inhalation: 1.35s (1.2-1.5s), ribcage pump-handle + bucket-handle expansion,
   *   spinal extension, and clavicle rise.
   * - Passive elastic exhalation: 2.25s (2.0-2.6s), exponential elastic recoil.
   * - Subtle 1.2Hz cardiac micro-pulse.
   */
  private applyBreathing(buf: PoseBuffer, t: number): void {
    const tInhale = 1.35; // 1.2 - 1.5s
    const tExhale = 2.25; // 2.0 - 2.6s
    const tCycle = tInhale + tExhale; // ~3.6s (16.7 breaths/min)

    const cycleTime = ((t % tCycle) + tCycle) % tCycle;
    let bRaw: number;

    if (cycleTime < tInhale) {
      // Phase 1: Active muscular inhalation via C2 minimum-jerk polynomial
      const tau = cycleTime / tInhale;
      bRaw = tau * tau * tau * (10 + tau * (-15 + 6 * tau));
    } else {
      // Phase 2: Passive elastic exhalation (exponential lung recoil)
      const tau = (cycleTime - tInhale) / tExhale;
      bRaw = (Math.exp(-2.8 * tau) - Math.exp(-2.8)) / (1.0 - Math.exp(-2.8));
    }

    // Natural tidal volume variation via slow noise modulation
    const b = bRaw * (1.0 + 0.12 * noise(t * 0.15, 41));

    // 1. Ribcage Pump-Handle Motion (AP diameter expansion -> chest & upperChest pitch up)
    buf.add('chest', 0.024 * b, 0, 0);
    buf.add('upperChest', 0.016 * b, 0, 0);
    buf.add('neck', -0.007 * b, 0, 0); // Cervical stabilization

    // 2. Ribcage Bucket-Handle Motion (transverse expansion & vertical lift)
    buf.offsetPos('chest', 0, 0.005 * b, 0.002 * b);
    buf.offsetPos('hips', 0, 0.0018 * b, 0);

    // 3. Spinal Extension
    buf.add('spine', 0.014 * b, 0, 0);

    // 4. Clavicle Rise
    buf.add('leftShoulder', 0, 0, 0.018 * b);
    buf.add('rightShoulder', 0, 0, -0.018 * b);

    // 5. Upper Arm Passive Expansion
    buf.add('leftUpperArm', 0.012 * b, 0, 0);
    buf.add('rightUpperArm', 0.012 * b, 0, 0);

    // 6. 1.2Hz Cardiac Micro-Pulse (72 bpm ballistocardiographic mechanical pulse)
    const heartPhase = t * 1.2 * 2 * Math.PI;
    const c = 0.0012 * (Math.sin(heartPhase) + 0.35 * Math.sin(2 * heartPhase));
    buf.add('chest', c, 0, 0);
    buf.add('upperChest', c * 0.7, 0, 0);
    buf.offsetPos('chest', 0, c * 0.15, 0);
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

  /**
   * Coordinated Full-Body Kinematic Chain:
   * Gaze leads -> Head (pitch/yaw) -> Neck (40ms phase lag) -> UpperChest -> Chest -> Spine.
   * Also drives Saccade-coupled blinking and spontaneous Poisson-distributed blinking (3-5s).
   */
  private applyLook(buf: PoseBuffer, dt: number, time: number): void {
    const idleFor = time - this.lastPointerTime;
    const relax = idleFor > 8 ? clamp((idleFor - 8) / 4, 0, 1) : 0;
    const goalX = this.targetLookX * (1 - relax);
    const goalY = this.targetLookY * (1 - relax);

    // 1. Gaze leads: fast target tracking
    const kGaze = 1 - Math.exp(-8 * dt);
    this.lookX += (goalX - this.lookX) * kGaze;
    this.lookY += (goalY - this.lookY) * kGaze;

    const gazeUp = (this.thinkingGaze || 0) * 0.5;
    const targetYaw = this.lookX * 0.55;
    const targetPitch = (this.lookY * 0.32 - gazeUp * 0.3);

    // 2. Coordinated Anatomical Kinematic Chain Propagation:
    // Head tracks gaze
    const kHead = 1 - Math.exp(-dt / 0.080);
    this.chainHead.yaw += (targetYaw - this.chainHead.yaw) * kHead;
    this.chainHead.pitch += (targetPitch - this.chainHead.pitch) * kHead;

    // Neck tracks Head with 40ms phase lag
    const kNeck = 1 - Math.exp(-dt / 0.040);
    this.chainNeck.yaw += (this.chainHead.yaw - this.chainNeck.yaw) * kNeck;
    this.chainNeck.pitch += (this.chainHead.pitch - this.chainNeck.pitch) * kNeck;

    // UpperChest tracks Neck
    const kUpperChest = 1 - Math.exp(-dt / 0.045);
    this.chainUpperChest.yaw += (this.chainNeck.yaw - this.chainUpperChest.yaw) * kUpperChest;
    this.chainUpperChest.pitch += (this.chainNeck.pitch - this.chainUpperChest.pitch) * kUpperChest;

    // Chest tracks UpperChest
    const kChest = 1 - Math.exp(-dt / 0.050);
    this.chainChest.yaw += (this.chainUpperChest.yaw - this.chainChest.yaw) * kChest;
    this.chainChest.pitch += (this.chainUpperChest.pitch - this.chainChest.pitch) * kChest;

    // Spine tracks Chest
    const kSpine = 1 - Math.exp(-dt / 0.055);
    this.chainSpine.yaw += (this.chainChest.yaw - this.chainSpine.yaw) * kSpine;
    this.chainSpine.pitch += (this.chainChest.pitch - this.chainSpine.pitch) * kSpine;

    // Apply distributed rotations across the kinematic column
    buf.add('head', 0.46 * this.chainHead.pitch, 0.44 * this.chainHead.yaw, -0.035 * this.chainHead.yaw);
    buf.add('neck', 0.22 * this.chainNeck.pitch, 0.20 * this.chainNeck.yaw, -0.015 * this.chainNeck.yaw);
    buf.add('upperChest', 0.12 * this.chainUpperChest.pitch, 0.12 * this.chainUpperChest.yaw, 0);
    buf.add('chest', 0.08 * this.chainChest.pitch, 0.08 * this.chainChest.yaw, 0);
    buf.add('spine', 0.05 * this.chainSpine.pitch, 0.05 * this.chainSpine.yaw, 0);

    // 3. Saccade-Coupled & Large Head Turn Blinking
    this.blinkTimer += dt;
    if (this.blinkTimer >= this.nextBlinkInterval) {
      this.triggerBlink(Math.random() < 0.15); // 15% spontaneous double-blink
    }

    // Check large rapid head turn angular velocity
    const safeDt = dt > 0.001 ? dt : 0.016;
    const headYawVel = Math.abs(this.chainHead.yaw - this.prevHeadYaw) / safeDt;
    if (headYawVel > 0.6 && this.blinkTimer > 1.2) {
      this.triggerBlink();
    }
    this.prevHeadYaw = this.chainHead.yaw;
    this.prevHeadPitch = this.chainHead.pitch;

    // 4. Saccade state update
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

        // Saccade-coupled blink on rapid saccadic sweeps
        const jumpDist = Math.hypot(s.targetX - s.startX, s.targetY - s.startY);
        if (jumpDist > 0.045 && this.blinkTimer > 1.0) {
          this.triggerBlink();
        }
      }
    }

    // 5. Thinking Gaze Aversion: eyes dart upward 18° - 26° (0.31 - 0.45 rad)
    const thinkingGazeAngle = (this.thinkingGaze || 0) * (0.32 + 0.08 * Math.sin(time * 0.5));

    // 6. Vestibulo-Ocular Reflex (VOR): counter-rotate gaze against head pitch/yaw
    const vorX = -this.chainHead.yaw * 0.4;
    const vorY = -this.chainHead.pitch * 0.4;

    this.eyeGaze.x = this.lookX + s.currentX + vorX;
    this.eyeGaze.y = this.lookY + s.currentY - thinkingGazeAngle + vorY;
  }

  /**
   * Applies active generative parametric IK actions (e.g. resting chin on hand, touching cheek,
   * covering mouth, hand on heart, embrace, etc.) with smooth C2 minimum-jerk blending.
   */
  private applyGenerativeAction(buf: PoseBuffer, dt: number): void {
    const act = this.generativeAction;
    if (!act.active) return;

    act.timer += dt;
    const tau = clamp(act.timer / act.duration, 0, 1);

    if (tau >= 1.0) {
      act.active = false;
      return;
    }

    // 4-Phase minimum-jerk trajectory profile:
    // Reach (0.0 -> 0.35), Hold (0.35 -> 0.70), Ease-Out Return (0.70 -> 1.0)
    let w: number;
    if (tau < 0.35) {
      const p = tau / 0.35;
      w = p * p * p * (10 + p * (-15 + 6 * p));
    } else if (tau < 0.70) {
      w = 1.0;
    } else {
      const p = (1.0 - tau) / 0.30;
      w = p * p * p * (10 + p * (-15 + 6 * p));
    }

    if (w <= 0.001) return;

    // Evaluate Analytical 2-Bone Arm IK and blend into PoseBuffer
    if (act.lateral === 'right' || act.lateral === 'both') {
      solveTwoBoneArmIK('right', act.rightTarget, act.outUpperR, act.outLowerR, 0.25, 0.22, act.rightPole);
      buf.blend('rightUpperArm', act.outUpperR, w);
      buf.blend('rightLowerArm', act.outLowerR, w);
      if (act.rightFingers) {
        buf.setFingers('right', act.rightFingers);
      }
    }

    if (act.lateral === 'left' || act.lateral === 'both') {
      solveTwoBoneArmIK('left', act.leftTarget, act.outUpperL, act.outLowerL, 0.25, 0.22, act.leftPole);
      buf.blend('leftUpperArm', act.outUpperL, w);
      buf.blend('leftLowerArm', act.outLowerL, w);
      if (act.leftFingers) {
        buf.setFingers('left', act.leftFingers);
      }
    }

    // Torso lean coupling
    if (act.lean !== 0) {
      buf.add('spine', act.lean * 0.10 * w, 0, 0);
      buf.add('chest', act.lean * 0.08 * w, 0, 0);
    }
  }
}
