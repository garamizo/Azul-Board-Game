const FILES = {
  select: 'select.wav', invalid: 'invalid.wav', yourTurn: 'yourTurn.wav', botMove: 'botMove.wav',
  score: 'score.wav', win: 'win.wav', lose: 'lose.mp3',
} as const;
export type SoundName = keyof typeof FILES;

const KEY = 'azul.muted';
function readMuted(): boolean {
  try { return localStorage.getItem(KEY) === '1'; } catch { return false; }
}

export const soundState = $state({ muted: readMuted() });

// Browsers play audio only after a user gesture.
let unlocked = false;
if (typeof window !== 'undefined') window.addEventListener('pointerdown', () => { unlocked = true; }, { once: true });

const cache = new Map<SoundName, HTMLAudioElement>();

export const sounds = {
  play(name: SoundName): void {
    if (soundState.muted || !unlocked) return;
    let audio = cache.get(name);
    if (!audio) {
      audio = new Audio(`/assets/sounds/${FILES[name]}`);
      cache.set(name, audio);
    }
    audio.currentTime = 0;
    void audio.play().catch(() => { /* autoplay refused */ });
  },
  toggle(): void {
    soundState.muted = !soundState.muted;
    try { localStorage.setItem(KEY, soundState.muted ? '1' : '0'); } catch { /* storage blocked */ }
  },
};
