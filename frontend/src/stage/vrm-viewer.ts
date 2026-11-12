import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { VRMLoaderPlugin, VRMUtils, type VRM } from '@pixiv/three-vrm';
import { ProceduralAnimator } from './procedural-animator.js';
import { LipSyncAnimator, type VisemeTickEvent } from './lip-sync.js';

const MAX_DELTA = 0.1;

export class VRMViewer {
  public container: HTMLElement | null = null;
  public scene: THREE.Scene | null = null;
  public camera: THREE.PerspectiveCamera | null = null;
  public renderer: THREE.WebGLRenderer | null = null;
  public controls: OrbitControls | null = null;
  public lookTarget = new THREE.Object3D();

  public currentVrm: VRM | null = null;
  public placeholderMesh: THREE.Group | null = null;
  public proceduralAnimator = new ProceduralAnimator();
  public lipSyncAnimator = new LipSyncAnimator();

  public state = 'idle';
  private userMovedCamera = false;
  private time = 0;
  private lastFrame = 0;
  private loadToken = 0;
  private fitDistance = 1.4;
  private animationFrameId: number | null = null;
  private resizeObserver: ResizeObserver | null = null;

  private onContextLost = (e: Event) => e.preventDefault();
  private onContextRestored = () => this.handleResize();

  public async init(containerElement: HTMLElement): Promise<void> {
    this.container = containerElement;
    this.scene = new THREE.Scene();
    this.scene.add(this.lookTarget);

    const { width, height } = this.getSize();

    this.camera = new THREE.PerspectiveCamera(30, width / height, 0.05, 30);
    this.camera.position.set(0, 1.3, 1.6);

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(width, height, false);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.NoToneMapping;

    this.renderer.domElement.addEventListener('webglcontextlost', this.onContextLost);
    this.renderer.domElement.addEventListener('webglcontextrestored', this.onContextRestored);
    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.enablePan = false;
    this.controls.rotateSpeed = 0.5;
    this.controls.minPolarAngle = Math.PI * 0.32;
    this.controls.maxPolarAngle = Math.PI * 0.58;
    this.controls.minAzimuthAngle = -Math.PI * 0.35;
    this.controls.maxAzimuthAngle = Math.PI * 0.35;
    this.controls.target.set(0, 1.2, 0);
    this.controls.update();

    this.controls.addEventListener('start', () => {
      this.userMovedCamera = true;
    });

    this.setupLighting();
    this.createPlaceholderAvatar();

    this.resizeObserver = new ResizeObserver(() => this.handleResize());
    this.resizeObserver.observe(this.container);

    this.lastFrame = performance.now();
    this.animate();
  }

  public getSize(): { width: number; height: number } {
    const width = this.container?.clientWidth || window.innerWidth || 1;
    const height = this.container?.clientHeight || window.innerHeight || 1;
    return { width: Math.max(width, 1), height: Math.max(height, 1) };
  }

  private setupLighting(): void {
    if (!this.scene) return;
    const key = new THREE.DirectionalLight(0xffffff, Math.PI);
    key.position.set(0.5, 1.1, 1.6);
    this.scene.add(key);

    const rim = new THREE.DirectionalLight(0x9db0ff, 0.5);
    rim.position.set(-1.6, 1.6, -1.8);
    this.scene.add(rim);

    const amb = new THREE.AmbientLight(0xffffff, 0.3);
    this.scene.add(amb);
  }

  private createPlaceholderAvatar(): void {
    if (!this.scene) return;
    this.removePlaceholder();
    const group = new THREE.Group();
    const orb = new THREE.Mesh(
      new THREE.IcosahedronGeometry(0.22, 2),
      new THREE.MeshStandardMaterial({
        color: 0xa8b4ff,
        roughness: 0.35,
        metalness: 0.3,
        emissive: 0x23295a,
        emissiveIntensity: 0.8,
      })
    );
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.34, 0.006, 8, 96),
      new THREE.MeshBasicMaterial({ color: 0xf1e3b5, transparent: true, opacity: 0.7 })
    );
    ring.rotation.x = Math.PI / 2.4;
    group.add(orb, ring);
    group.position.set(0, 1.25, 0);
    this.placeholderMesh = group;
    this.scene.add(group);
  }

  private removePlaceholder(): void {
    if (!this.placeholderMesh || !this.scene) return;
    this.scene.remove(this.placeholderMesh);
    VRMUtils.deepDispose(this.placeholderMesh);
    this.placeholderMesh = null;
  }

  public async loadModel(urlOrBuffer: string | ArrayBuffer): Promise<VRM> {
    if (!this.scene) throw new Error('Viewer not initialized');
    const token = ++this.loadToken;
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));

    const gltf = await new Promise<any>((resolve, reject) => {
      if (typeof urlOrBuffer === 'string') {
        loader.load(urlOrBuffer, resolve, undefined, reject);
      } else {
        loader.parse(urlOrBuffer, '', resolve, reject);
      }
    });

    if (token !== this.loadToken) {
      if (gltf.userData?.vrm) VRMUtils.deepDispose(gltf.userData.vrm.scene);
      throw new Error('Superseded by a newer model load');
    }

    const vrm = gltf.userData.vrm as VRM;
    if (!vrm) throw new Error('File is not a valid VRM model');

    VRMUtils.removeUnnecessaryVertices?.(gltf.scene);
    VRMUtils.combineSkeletons?.(gltf.scene);

    vrm.scene.traverse((obj) => {
      obj.frustumCulled = false;
    });

    if (this.currentVrm) {
      this.scene.remove(this.currentVrm.scene);
      VRMUtils.deepDispose(this.currentVrm.scene);
      this.currentVrm = null;
    }
    this.removePlaceholder();

    VRMUtils.rotateVRM0(vrm);
    vrm.scene.position.set(0, 0, 0);
    if (vrm.lookAt) {
      vrm.lookAt.target = this.lookTarget;
      vrm.lookAt.autoUpdate = true;
    }

    this.currentVrm = vrm;
    this.proceduralAnimator.reset();
    this.lipSyncAnimator.reset();
    this.scene.add(vrm.scene);
    vrm.scene.updateMatrixWorld(true);
    vrm.springBoneManager?.reset?.();

    this.fitCameraToModel(vrm);
    return vrm;
  }

  public fitCameraToModel(vrm: VRM | null = this.currentVrm): void {
    if (!vrm || !this.camera || !this.controls) return;
    vrm.scene.updateMatrixWorld(true);

    const head = new THREE.Vector3();
    const node = vrm.humanoid?.getNormalizedBoneNode('head') || vrm.humanoid?.getRawBoneNode('head');
    if (node) {
      node.getWorldPosition(head);
    }

    if (!node || head.y < 0.3) {
      const box = new THREE.Box3().setFromObject(vrm.scene);
      const size = box.getSize(new THREE.Vector3());
      box.getCenter(head);
      head.y = box.min.y + size.y * 0.9;
    }

    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const span = 1.12;
    const widthSpan = 0.8;
    const byHeight = span / 2 / Math.tan(fov / 2);
    const byWidth = widthSpan / 2 / (Math.tan(fov / 2) * this.camera.aspect);
    this.fitDistance = Math.max(byHeight, byWidth, 0.8);

    const targetY = head.y - 0.17;
    this.controls.target.set(head.x, targetY, head.z);
    this.camera.position.set(head.x, targetY + 0.04, head.z + this.fitDistance);
    this.controls.minDistance = this.fitDistance * 0.55;
    this.controls.maxDistance = this.fitDistance * 1.8;
    this.controls.update();
  }

  public setEmotion(name: string, weight = 1, hold = 5): boolean {
    return this.proceduralAnimator.setEmotion(name, weight, hold);
  }

  public setState(state: string): void {
    this.state = state;
    this.lipSyncAnimator.setSpeaking(state === 'speaking');
    this.proceduralAnimator.setState(state);
  }

  public setSpeaking(speaking: boolean): void {
    if (speaking) {
      this.setState('speaking');
    } else if (this.state === 'speaking') {
      this.setState('idle');
    }
  }

  public playGesture(name: string, duration?: number): boolean {
    return this.proceduralAnimator.playGesture(name, duration);
  }

  public onVisemeTick(payload: VisemeTickEvent): void {
    this.lipSyncAnimator.onVisemeTick(payload);
  }

  public updateCursor(nx: number, ny: number): void {
    this.proceduralAnimator.setLookTarget(nx, ny, this.time);
    if (this.placeholderMesh) {
      this.placeholderMesh.rotation.y = nx * 0.35;
      this.placeholderMesh.rotation.x = ny * 0.2;
    }
  }

  public releaseCursor(): void {
    this.proceduralAnimator.setLookTarget(0, 0, this.time);
  }

  public resetCamera(): void {
    this.userMovedCamera = false;
    if (this.currentVrm) {
      this.fitCameraToModel(this.currentVrm);
    } else if (this.camera && this.controls) {
      this.camera.position.set(0, 1.3, 1.6);
      this.controls.target.set(0, 1.2, 0);
      this.controls.update();
    }
  }

  public handleResize(): void {
    if (!this.container || !this.renderer || !this.camera) return;
    const { width, height } = this.getSize();
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(width, height, false);
    if (this.currentVrm && !this.userMovedCamera) {
      this.fitCameraToModel(this.currentVrm);
    }
  }

  private updateLookTarget(): void {
    const gaze = this.proceduralAnimator.eyeGaze;
    if (!gaze || !this.camera) return;
    const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 0);
    const up = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 1);
    this.lookTarget.position
      .copy(this.camera.position)
      .addScaledVector(right, gaze.x * 0.45)
      .addScaledVector(up, -gaze.y * 0.3);
    this.lookTarget.updateMatrixWorld(true);
  }

  public animate(): void {
    this.animationFrameId = requestAnimationFrame(() => this.animate());

    const now = performance.now();
    const dt = Math.min((now - this.lastFrame) / 1000, MAX_DELTA);
    this.lastFrame = now;
    this.time += dt;

    this.controls?.update();

    if (this.currentVrm) {
      this.proceduralAnimator.update(this.currentVrm, dt, this.time, { mouth: this.lipSyncAnimator.mouthEnergy });
      this.lipSyncAnimator.update(this.currentVrm, dt);
      this.updateLookTarget();
      this.currentVrm.update(dt);
    } else if (this.placeholderMesh) {
      this.placeholderMesh.position.y = 1.25 + Math.sin(this.time * 1.6) * 0.02;
      this.placeholderMesh.children[1].rotation.z += dt * 0.4;
    }

    if (this.scene && this.camera && this.renderer) {
      this.renderer.render(this.scene, this.camera);
    }
  }

  public dispose(): void {
    if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);
    this.resizeObserver?.disconnect();
    this.controls?.dispose();
    if (this.currentVrm) VRMUtils.deepDispose(this.currentVrm.scene);
    this.removePlaceholder();
    if (this.renderer) {
      this.renderer.domElement.removeEventListener('webglcontextlost', this.onContextLost);
      this.renderer.domElement.removeEventListener('webglcontextrestored', this.onContextRestored);
      this.renderer.dispose();
      this.renderer.domElement.remove();
    }
  }
}
