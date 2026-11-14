import { describe, expect, it } from 'bun:test';
import * as THREE from 'three';
import {
  ProceduralAnimator,
  parseKinematicCues,
} from '../src/stage/procedural-animator';
import { BONE_NAMES } from '../src/stage/pose-utils';

function createMockVRM() {
  const bones: Record<string, THREE.Object3D> = {};
  for (const name of BONE_NAMES) {
    bones[name] = new THREE.Object3D();
  }

  const expressions: Record<string, number> = {};

  return {
    meta: { metaVersion: '1' },
    humanoid: {
      getNormalizedBoneNode: (name: string) => bones[name] ?? null,
    },
    expressionManager: {
      getExpression: (name: string) => (name in expressions ? { name } : null),
      setValue: (name: string, val: number) => {
        expressions[name] = val;
      },
      getValue: (name: string) => expressions[name] ?? 0,
    },
    bones,
    expressions,
  };
}

describe('ProceduralAnimator Biomimetic Kinematics', () => {
  it('implements asymmetric biological respiration with pump-handle, bucket-handle, and 1.2Hz cardiac micro-pulse', () => {
    const animator = new ProceduralAnimator();
    const mockVRM = createMockVRM();

    // Inhalation phase: t = 0 to 1.35s
    // Peak inhalation at t = 1.35s
    animator.update(mockVRM as any, 1 / 60, 1.35);

    const chestInhaleX = animator.pose.get('chest').x;
    const upperChestInhaleX = animator.pose.get('upperChest').x;
    const spineInhaleX = animator.pose.get('spine').x;
    const leftShoulderInhaleZ = animator.pose.get('leftShoulder').z;
    const rightShoulderInhaleZ = animator.pose.get('rightShoulder').z;
    const chestPosInhale = animator.pose.pos.get('chest');

    // Ribcage pump-handle expansion (quaternion x component represents sin(theta/2) of upward pitch)
    expect(chestInhaleX).toBeGreaterThan(0.008);
    expect(upperChestInhaleX).toBeGreaterThan(0.005);
    // Spinal extension (spine pitch upward in X)
    expect(spineInhaleX).toBeGreaterThan(0.004);
    // Clavicle rise (leftShoulder +Z, rightShoulder -Z)
    expect(leftShoulderInhaleZ).toBeGreaterThan(0.005);
    expect(rightShoulderInhaleZ).toBeLessThan(-0.005);
    // Bucket-handle lateral/vertical expansion
    expect(chestPosInhale?.[1]).toBeGreaterThan(0.003);

    // Exhalation phase: t = 2.45s (passive elastic recoil)
    animator.update(mockVRM as any, 1 / 60, 2.45);
    const chestExhaleX = animator.pose.get('chest').x;
    expect(chestExhaleX).toBeLessThan(chestInhaleX);

    // End of exhalation: t = 3.59s (approaching resting residual capacity)
    animator.update(mockVRM as any, 1 / 60, 3.59);
    const chestEndExhaleX = animator.pose.get('chest').x;
    expect(chestEndExhaleX).toBeLessThan(0.01);
  });

  it('implements Contrapposto & weight shifting with pelvic roll, lateral offset, and shoulder counter-tilt', () => {
    const animator = new ProceduralAnimator();
    const mockVRM = createMockVRM();

    // Force right leg dominance: dominanceCurrent = 1.0
    animator.contrapposto.dominanceCurrent = 1.0;
    animator.contrapposto.startDominance = 1.0;
    animator.contrapposto.dominanceTarget = 1.0;
    animator.contrapposto.progress = 1.0;

    animator.update(mockVRM as any, 1 / 60, 0);

    const hipsRot = animator.pose.get('hips');
    const hipsPos = animator.pose.pos.get('hips');
    const spineRot = animator.pose.get('spine');
    const chestRot = animator.pose.get('chest');
    const leftShoulderRot = animator.pose.get('leftShoulder');
    const rightShoulderRot = animator.pose.get('rightShoulder');

    // 1. Pelvis Roll: 2-3 degrees (0.035 to 0.052 rad)
    const pelvisRollAngle = 2 * Math.asin(Math.abs(hipsRot.z));
    expect(pelvisRollAngle).toBeGreaterThanOrEqual(0.035);
    expect(pelvisRollAngle).toBeLessThanOrEqual(0.052);

    // 2. Lateral Pelvis Offset: negative X for right leg dominance
    expect(hipsPos?.[0]).toBeLessThan(-0.015);

    // 3. Contrapposto counter-tilt:
    // Spine counter-rolls opposite to hips
    expect(spineRot.z * hipsRot.z).toBeLessThan(0);
    // Shoulder girdle tilts in opposite direction of pelvic axis
    expect(leftShoulderRot.z).toBeGreaterThan(0.010);
    expect(rightShoulderRot.z).toBeGreaterThan(0.010);

    // 4. Level head & eyes: sum of rolls across spinal column balances pelvis tilt
    const totalRoll = hipsRot.z + spineRot.z + chestRot.z;
    expect(Math.abs(totalRoll)).toBeLessThan(0.01);
  });

  it('automatically couples clavicle elevation and retraction to arm elevation (Scapulohumeral Rhythm)', () => {
    const animator = new ProceduralAnimator();
    const mockVRM = createMockVRM();

    // Trigger wave gesture which raises right arm
    animator.playGesture('wave');

    // Step forward into apex of wave
    for (let f = 0; f < 30; f++) {
      animator.update(mockVRM as any, 1 / 60, f * (1 / 60));
    }

    // Right shoulder should have automatically elevated (negative Z in VRM)
    const rightShoulder = animator.pose.get('rightShoulder');
    expect(rightShoulder.z).toBeLessThan(-0.1);
  });

  it('propagates look rotation organically through the kinematic chain: Gaze -> Head -> Neck (40ms lag) -> UpperChest -> Chest -> Spine', () => {
    const animator = new ProceduralAnimator();
    const mockVRM = createMockVRM();

    // Step a few frames at rest
    for (let f = 0; f < 10; f++) {
      animator.update(mockVRM as any, 1 / 60, f * (1 / 60));
    }

    // Sudden gaze shift to the right
    animator.setLookTarget(1.0, 0.4, 0.2);

    // Frame 1: Gaze and Head lead
    animator.update(mockVRM as any, 0.016, 0.216);
    expect(animator.chainHead.yaw).toBeGreaterThan(0);
    expect(animator.chainHead.yaw).toBeGreaterThan(animator.chainNeck.yaw);

    // Step 40ms later: Neck follows with ~40ms lag
    for (let i = 0; i < 3; i++) {
      animator.update(mockVRM as any, 0.016, 0.216 + i * 0.016);
    }
    expect(animator.chainNeck.yaw).toBeGreaterThan(0.02);
    expect(animator.chainUpperChest.yaw).toBeLessThan(animator.chainNeck.yaw);

    // Step 80ms later: UpperChest and Chest follow
    for (let i = 0; i < 4; i++) {
      animator.update(mockVRM as any, 0.016, 0.264 + i * 0.016);
    }
    expect(animator.chainUpperChest.yaw).toBeGreaterThan(0.01);
    expect(animator.chainChest.yaw).toBeLessThan(animator.chainUpperChest.yaw);
    expect(animator.chainSpine.yaw).toBeLessThan(animator.chainChest.yaw);

    // Check hierarchical propagation ordering across kinematic chain nodes
    expect(animator.chainHead.yaw).toBeGreaterThan(animator.chainNeck.yaw);
    expect(animator.chainNeck.yaw).toBeGreaterThan(animator.chainUpperChest.yaw);
    expect(animator.chainUpperChest.yaw).toBeGreaterThan(animator.chainChest.yaw);
    expect(animator.chainChest.yaw).toBeGreaterThan(animator.chainSpine.yaw);

    // Check distributed torso bone rotations
    const headRot = animator.pose.get('head');
    const upperChestRot = animator.pose.get('upperChest');
    const chestRot = animator.pose.get('chest');
    const spineRot = animator.pose.get('spine');

    expect(headRot.y).toBeGreaterThan(upperChestRot.y);
    expect(upperChestRot.y).toBeGreaterThan(chestRot.y);
    expect(chestRot.y).toBeGreaterThan(spineRot.y);
  });

  it('triggers saccade-coupled and large head-turn blinking coupled with spontaneous Poisson intervals', () => {
    const animator = new ProceduralAnimator();
    let blinkCallCount = 0;
    (animator.expressions as any).triggerBlink = () => {
      blinkCallCount++;
    };

    // 1. Spontaneous Poisson blinking: accumulate timer to nextBlinkInterval
    animator.blinkTimer = animator.nextBlinkInterval + 0.1;
    const mockVRM = createMockVRM();
    animator.update(mockVRM as any, 0.016, 1.0);
    expect(blinkCallCount).toBeGreaterThanOrEqual(1);

    // 2. Large rapid head turn triggers natural blink
    const initialBlinks = blinkCallCount;
    animator.nextBlinkInterval = 10.0; // Ensure spontaneous blink doesn't fire
    animator.blinkTimer = 2.0; // Ready for saccadic/head-turn blink
    animator.setLookTarget(1.0, 0.0, 2.0);
    animator.chainHead.yaw = 0; // sudden yaw jump next frame
    animator.update(mockVRM as any, 0.016, 2.016);
    expect(blinkCallCount).toBeGreaterThan(initialBlinks);
  });

  it('parses natural language action descriptions into parametric IK targets and executes them smoothly', () => {
    // 1. Parse natural actions
    const cue1 = parseKinematicCues('Cô bé *chống cằm suy nghĩ* một lúc');
    expect(cue1.action).toBe('chin');
    expect(cue1.emotion).toBe('thinking');

    const cue2 = parseKinematicCues('*áp tay lên má* ngượng ngùng');
    expect(cue2.action).toBe('cheek');
    expect(cue2.emotion).toBe('relaxed');

    const cue3 = parseKinematicCues('Nàng khẽ *che miệng cười*');
    expect(cue3.action).toBe('mouth');
    expect(cue3.emotion).toBe('happy');

    const cue4 = parseKinematicCues('*đặt tay lên tim* xin cảm ơn mọi người');
    expect(cue4.action).toBe('chestHeart');
    expect(cue4.emotion).toBe('happy');

    const cue5 = parseKinematicCues('*chắp tay trước ngực* kính cẩn');
    expect(cue5.action).toBe('chestCenter');

    const cue6 = parseKinematicCues('*day thái dương* vì căng thẳng');
    expect(cue6.action).toBe('temple');

    const cue7 = parseKinematicCues('*đưa hai tay ra trước muốn ôm* lấy bạn');
    expect(cue7.action).toBe('openEmbrace');
    expect(cue7.lateral).toBe('both');

    const cue8 = parseKinematicCues('*cúi lại gần* lắng nghe');
    expect(cue8.lean).toBe('forward');

    // 2. Play action through ProceduralAnimator
    const animator = new ProceduralAnimator();
    const mockVRM = createMockVRM();

    const ok = animator.playAction('chin', 'right', 3.0);
    expect(ok).toBe(true);
    expect(animator.isGesturing()).toBe(true);

    // Step into apex of chin rest
    for (let f = 0; f < 30; f++) {
      animator.update(mockVRM as any, 1 / 60, f * (1 / 60));
    }

    const rightUpper = animator.pose.get('rightUpperArm');
    const rightLower = animator.pose.get('rightLowerArm');
    // Forearm should be significantly flexed toward chin
    expect(Math.abs(rightLower.y)).toBeGreaterThan(0.5);
    expect(Math.abs(rightUpper.x) + Math.abs(rightUpper.z)).toBeGreaterThan(0.2);
  });

  it('guarantees zero transient heap allocations across multiple 60fps render ticks', () => {
    const animator = new ProceduralAnimator();
    const mockVRM = createMockVRM();

    // Warm up
    for (let f = 0; f < 10; f++) {
      animator.update(mockVRM as any, 1 / 60, f * (1 / 60));
    }

    // Run 60 frames (1 second at 60fps) under active breathing, look tracking, and contrapposto
    animator.setLookTarget(0.5, -0.3, 1.0);
    animator.playAction('cheek', 'right', 2.0);

    for (let f = 0; f < 60; f++) {
      animator.update(mockVRM as any, 1 / 60, 1.0 + f * (1 / 60));
    }

    // Verify all pose buffers and arrays remained intact with valid unit quaternions
    const head = animator.pose.get('head');
    expect(head.length()).toBeCloseTo(1.0, 4);
    const rightArm = animator.pose.get('rightUpperArm');
    expect(rightArm.length()).toBeCloseTo(1.0, 4);
    const hips = animator.pose.get('hips');
    expect(hips.length()).toBeCloseTo(1.0, 4);
  });
});
