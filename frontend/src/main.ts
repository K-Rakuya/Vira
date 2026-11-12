import './style.css';
import { Starfield } from './stage/starfield.js';
import { VRMViewer } from './stage/vrm-viewer.js';
import { HudController } from './ui/hud-controller.js';

async function bootstrap() {
  // 1. Start cosmic starfield background
  const starfieldCanvas = document.getElementById('starfield') as HTMLCanvasElement;
  if (starfieldCanvas) {
    const starfield = new Starfield(starfieldCanvas);
    starfield.start();
  }

  // 2. Initialize 3D VRM Stage
  const container = document.getElementById('canvas-container');
  const stageError = document.getElementById('stage-error');
  const vrmViewer = new VRMViewer();

  if (container) {
    try {
      await vrmViewer.init(container);
    } catch (err) {
      console.error('Failed to initialize 3D WebGL context:', err);
      if (stageError) stageError.hidden = false;
    }
  }

  // 3. Initialize Glassmorphic HUD Controller
  const hud = new HudController(vrmViewer);
  await hud.init();

  // 4. Load initial VRM model
  try {
    await vrmViewer.loadModel('/models/avatar.vrm');
  } catch (err) {
    console.warn('Initial avatar load notice:', err);
  }

  // 5. Cleanup hooks
  window.addEventListener('beforeunload', () => {
    hud.dispose();
    vrmViewer.dispose();
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', bootstrap);
} else {
  bootstrap();
}
