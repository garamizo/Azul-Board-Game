// Copies the desktop game's sprites and sounds into public/assets.
import { cpSync, mkdirSync } from 'node:fs';

const src = new URL('../../assets/', import.meta.url);
const dst = new URL('../public/assets/', import.meta.url);
mkdirSync(new URL('sprites/', dst), { recursive: true });
mkdirSync(new URL('sounds/', dst), { recursive: true });

const sprites = ['board2.png', 'factory.png', 'tile_blue.png', 'tile_yellow.png', 'tile_red.png',
  'tile_black.png', 'tile_white.png', 'tile_first.png'];
for (const f of sprites) cpSync(new URL(`sprites/${f}`, src), new URL(`sprites/${f}`, dst));

// name in the web app -> file in assets/sounds (the desktop game's choices, azul/game.py)
const sounds = {
  select: 'mixkit-poker-card-placement-2001.wav',
  invalid: 'mixkit-video-game-mystery-alert-234.wav',
  yourTurn: 'mixkit-paper-slide-1530.wav',
  botMove: 'mixkit-retro-confirmation-tone-2860.wav',
  score: 'mixkit-small-win-2020.wav',
  win: 'mixkit-medieval-show-fanfare-announcement-226.wav',
  lose: 'Bidibodi_bidibu_radio.mp3',
};
for (const [name, file] of Object.entries(sounds)) {
  cpSync(new URL(`sounds/${file}`, src), new URL(`sounds/${name}${file.slice(file.lastIndexOf('.'))}`, dst));
}
