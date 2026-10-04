export class GhostAnimation extends HTMLElement {
  play(): void;
  pause(): void;
  restart(): void;
  wink(duration?: number): void;
  readonly paused: boolean;
}
