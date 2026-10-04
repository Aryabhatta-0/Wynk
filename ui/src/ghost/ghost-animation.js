import { ghostFrames } from './ghost-frames.js';

/** A standalone, responsive animation. No video playback or runtime dependencies. */
export class GhostAnimation extends HTMLElement {
  static observedAttributes = ['paused', 'speed'];

  #canvas;
  #context;
  #observer;
  #motion = matchMedia('(prefers-reduced-motion: reduce)');
  #elapsed = 0;
  #previousTime = null;
  #request = null;
  #width = 0;
  #height = 0;

  constructor() {
    super();
    const shadow = this.attachShadow({ mode: 'open' });
    shadow.innerHTML = '<style>:host{display:block}canvas{display:block;width:100%;height:100%}</style><canvas aria-hidden="true"></canvas>';
    this.#canvas = shadow.querySelector('canvas');
    this.#context = this.#canvas.getContext('2d');
  }

  get paused() { return this.hasAttribute('paused'); }
  get speed() {
    const value = Number(this.getAttribute('speed') ?? 1);
    return Number.isFinite(value) && value > 0 ? value : 1;
  }

  connectedCallback() {
    if (this.#motion.matches) this.setAttribute('paused', '');
    this.#observer = new ResizeObserver(this.#resize);
    this.#observer.observe(this);
    document.addEventListener('visibilitychange', this.#visibility);
    this.#motion.addEventListener('change', this.#motionChange);
    this.#schedule();
  }

  disconnectedCallback() {
    this.#stop();
    this.#observer?.disconnect();
    document.removeEventListener('visibilitychange', this.#visibility);
    this.#motion.removeEventListener('change', this.#motionChange);
  }

  attributeChangedCallback(name) {
    if (name === 'paused') {
      this.#stop();
      this.#schedule();
      this.dispatchEvent(new Event('playbackchange'));
    }
  }

  play() { this.removeAttribute('paused'); }
  pause() { this.setAttribute('paused', ''); }
  restart() {
    this.#elapsed = 0;
    this.#previousTime = null;
    this.#draw();
    this.play();
  }

  #motionChange = (event) => { if (event.matches) this.pause(); };
  #visibility = () => {
    this.#stop();
    this.#schedule();
  };

  #stop() {
    if (this.#request !== null) cancelAnimationFrame(this.#request);
    this.#request = null;
    this.#previousTime = null;
  }

  #schedule() {
    if (this.isConnected && !this.paused && !document.hidden && this.#request === null) {
      this.#request = requestAnimationFrame(this.#tick);
    }
  }

  #tick = (time) => {
    this.#request = null;
    if (this.#previousTime !== null) this.#elapsed += Math.min(time - this.#previousTime, 100) * this.speed;
    this.#previousTime = time;
    this.#draw();
    this.#schedule();
  };

  #resize = () => {
    const { width, height } = this.getBoundingClientRect();
    this.#width = width;
    this.#height = height;
    const ratio = Math.min(devicePixelRatio || 1, 2);
    this.#canvas.width = Math.round(width * ratio);
    this.#canvas.height = Math.round(height * ratio);
    this.#context.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.#draw();
  };

  #path(first, second, mix) {
    const context = this.#context;
    const count = first.length / 2;
    const x = (index) => first[index * 2] + (second[index * 2] - first[index * 2]) * mix;
    const y = (index) => first[index * 2 + 1] + (second[index * 2 + 1] - first[index * 2 + 1]) * mix;
    context.beginPath();
    context.moveTo((x(count - 1) + x(0)) / 2, (y(count - 1) + y(0)) / 2);
    for (let index = 0; index < count; index++) {
      const next = (index + 1) % count;
      context.quadraticCurveTo(x(index), y(index), (x(index) + x(next)) / 2, (y(index) + y(next)) / 2);
    }
    context.closePath();
  }

  #draw() {
    if (!this.#width || !this.#height) return;
    const context = this.#context;
    const { frames, fps, width, height } = ghostFrames;
    const duration = (frames.length - 1) * 1000 / fps;
    // Play the reference forward, then return through the same poses. Ease just
    // the turnarounds so the incomplete source clip becomes a continuous loop.
    const phase = (this.#elapsed % (duration * 2)) / duration;
    const progress = phase <= 1 ? phase : 2 - phase;
    const edge = 0.12;
    let eased;
    if (progress < edge) eased = progress * progress / (2 * edge * (1 - edge));
    else if (progress > 1 - edge) eased = 1 - (1 - progress) ** 2 / (2 * edge * (1 - edge));
    else eased = (progress - edge / 2) / (1 - edge);
    const position = eased * (frames.length - 1);
    const index = Math.min(Math.floor(position), frames.length - 2);
    const mix = position - index;
    const first = frames[index];
    const second = frames[index + 1];

    context.clearRect(0, 0, this.#width, this.#height);
    context.save();
    const scale = Math.min(this.#width / width, this.#height / height) * .87;
    context.translate((this.#width - width * scale) / 2, (this.#height - height * scale) / 2);
    context.scale(scale, scale);

    this.#path(first.body, second.body, mix);
    context.save();
    context.clip();
    const bounds = first.body.filter((_, i) => i % 2 === 0);
    const center = (Math.min(...bounds) + Math.max(...bounds)) / 2;
    const light = context.createRadialGradient(center - 90, 240, 20, center - 30, 300, 460);
    light.addColorStop(0, '#f8f9f4');
    light.addColorStop(.48, '#f1f2ee');
    light.addColorStop(1, '#dedfdb');
    context.fillStyle = light;
    context.fillRect(0, 0, width, height);
    context.restore();

    context.fillStyle = '#151519';
    first.eyes.forEach((eye, eyeIndex) => {
      this.#path(eye, second.eyes[eyeIndex], mix);
      context.fill();
    });
    context.restore();
  }
}

if (!customElements.get('ghost-animation')) customElements.define('ghost-animation', GhostAnimation);
