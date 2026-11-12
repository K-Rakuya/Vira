const PALETTE = ['255,255,255', '214,224,255', '255,240,214', '188,204,255'];

interface Star {
  x: number;
  y: number;
  depth: number;
  r: number;
  alpha: number;
  speed: number;
  phase: number;
  color: string;
}

interface ShootingStar {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  max: number;
  length: number;
}

export class Starfield {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D | null;
  private density: number;
  private stars: Star[] = [];
  private shooting: ShootingStar | null = null;
  private nextShoot: number;
  private pointer = { x: 0, y: 0, sx: 0, sy: 0 };
  private raf = 0;
  private time = 0;
  private last = 0;
  private width = 0;
  private height = 0;
  private reduceMotion = false;
  private motionQuery: MediaQueryList | null = null;

  constructor(canvas: HTMLCanvasElement, { density = 1 / 5200 }: { density?: number } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.density = density;
    this.nextShoot = 6 + Math.random() * 6;

    if (typeof window !== 'undefined' && window.matchMedia) {
      this.motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    }
  }

  private onResize = () => this.resize();
  private onPointer = (e: PointerEvent) => {
    this.pointer.x = e.clientX / Math.max(window.innerWidth, 1) - 0.5;
    this.pointer.y = e.clientY / Math.max(window.innerHeight, 1) - 0.5;
  };
  private onVisibility = () => (document.hidden ? this.pause() : this.resume());
  private onMotionChange = () => {
    this.reduceMotion = Boolean(this.motionQuery?.matches);
    this.pause();
    if (this.reduceMotion) {
      this.drawFrame(0);
    } else {
      this.resume();
    }
  };

  public start(): void {
    if (!this.ctx) return;
    this.reduceMotion = Boolean(this.motionQuery?.matches);
    this.resize();
    window.addEventListener('resize', this.onResize);
    window.addEventListener('pointermove', this.onPointer, { passive: true });
    document.addEventListener('visibilitychange', this.onVisibility);
    this.motionQuery?.addEventListener?.('change', this.onMotionChange);

    if (this.reduceMotion) {
      this.drawFrame(0);
    } else {
      this.resume();
    }
  }

  public stop(): void {
    this.pause();
    window.removeEventListener('resize', this.onResize);
    window.removeEventListener('pointermove', this.onPointer);
    document.removeEventListener('visibilitychange', this.onVisibility);
    this.motionQuery?.removeEventListener?.('change', this.onMotionChange);
  }

  public resume(): void {
    if (this.raf || this.reduceMotion) return;
    this.last = performance.now();
    const loop = (now: number) => {
      const dt = Math.min((now - this.last) / 1000, 0.1);
      this.last = now;
      this.time += dt;
      this.drawFrame(dt);
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  public pause(): void {
    if (this.raf) {
      cancelAnimationFrame(this.raf);
      this.raf = 0;
    }
  }

  public resize(): void {
    if (!this.ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    this.canvas.width = Math.floor(this.width * dpr);
    this.canvas.height = Math.floor(this.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const count = Math.max(90, Math.min(650, Math.round(this.width * this.height * this.density)));
    this.stars = Array.from({ length: count }, () => {
      const depth = Math.random() ** 1.8;
      return {
        x: Math.random(),
        y: Math.random(),
        depth,
        r: 0.35 + depth * 1.15,
        alpha: 0.28 + Math.random() * 0.6,
        speed: 0.5 + Math.random() * 1.8,
        phase: Math.random() * Math.PI * 2,
        color: PALETTE[Math.floor(Math.random() * PALETTE.length)],
      };
    });

    if (this.reduceMotion) {
      this.drawFrame(0);
    }
  }

  private spawnShootingStar(): void {
    const fromLeft = Math.random() < 0.5;
    this.shooting = {
      x: this.width * (fromLeft ? Math.random() * 0.5 : 0.5 + Math.random() * 0.5),
      y: this.height * Math.random() * 0.4,
      vx: (fromLeft ? 1 : -1) * (520 + Math.random() * 220),
      vy: 220 + Math.random() * 120,
      life: 0,
      max: 0.9 + Math.random() * 0.4,
      length: 110 + Math.random() * 90,
    };
  }

  private drawFrame(dt: number): void {
    const ctx = this.ctx;
    if (!ctx) return;

    ctx.clearRect(0, 0, this.width, this.height);

    const k = 1 - Math.exp(-3 * dt);
    this.pointer.sx += (this.pointer.x - this.pointer.sx) * k;
    this.pointer.sy += (this.pointer.y - this.pointer.sy) * k;

    for (const s of this.stars) {
      const drift = this.reduceMotion ? 0 : this.time * (0.0025 + s.depth * 0.006);
      let x = (s.x + drift) % 1;
      if (x < 0) x += 1;
      const px = x * this.width - this.pointer.sx * s.depth * 26;
      const py = s.y * this.height - this.pointer.sy * s.depth * 16;
      const twinkle = this.reduceMotion ? 1 : 0.62 + 0.38 * Math.sin(this.time * s.speed + s.phase);
      const a = s.alpha * twinkle;

      ctx.fillStyle = `rgba(${s.color},${a})`;
      ctx.beginPath();
      ctx.arc(px, py, s.r, 0, Math.PI * 2);
      ctx.fill();

      if (s.r > 1.1) {
        ctx.fillStyle = `rgba(${s.color},${a * 0.14})`;
        ctx.beginPath();
        ctx.arc(px, py, s.r * 3.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    if (this.reduceMotion) return;

    this.nextShoot -= dt;
    if (!this.shooting && this.nextShoot <= 0) {
      this.spawnShootingStar();
      this.nextShoot = 8 + Math.random() * 10;
    }

    const m = this.shooting;
    if (m) {
      m.life += dt;
      m.x += m.vx * dt;
      m.y += m.vy * dt;
      const t = m.life / m.max;
      if (t >= 1) {
        this.shooting = null;
        return;
      }
      const fade = Math.sin(Math.PI * t);
      const mag = Math.hypot(m.vx, m.vy);
      const tx = m.x - (m.vx / mag) * m.length;
      const ty = m.y - (m.vy / mag) * m.length;
      const g = ctx.createLinearGradient(m.x, m.y, tx, ty);
      g.addColorStop(0, `rgba(255,255,255,${0.85 * fade})`);
      g.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.strokeStyle = g;
      ctx.lineWidth = 1.4;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(m.x, m.y);
      ctx.lineTo(tx, ty);
      ctx.stroke();
    }
  }
}
