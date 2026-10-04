import type { GameView } from './types';
import { COLOR_NAMES } from './geometry';
import { seatName } from './names';

export interface Narration { who: string | null; text: string; color?: number }

const FIRST = 5;
const signed = (n: number) => (n > 0 ? ` (+${n})` : n < 0 ? ` (−${-n})` : '');

/// What changed from `prev` to `next`, as bubbles. Parts inferred by comparing
/// the two views (the marker leaving the centre, a score change) are added only
/// when `next` is the very next version: the event stream coalesces, so after a
/// gap they could belong to a different move. Skipped moves are not narrated.
export function narrate(prev: GameView, next: GameView): Narration[] {
  const before = prev.board, after = next.board;
  if (!before || !after) return [];
  const out: Narration[] = [];
  const consecutive = next.version === prev.version + 1;
  const m = next.lastMove;
  if (m && m.version === next.version && narrated(prev, next, m.seat)) {
    const who = seatName(next, m.seat);
    if (m.kind === 'take') {
      if (m.color === FIRST) {
        out.push({ who, text: 'took the first-player marker', color: FIRST });
      } else {
        const centre = m.factory === after.factories.length;
        const from = centre ? 'the centre' : `factory ${m.factory! + 1}`;
        const to = m.row === 5 ? 'floor' : `line ${m.row! + 1}`;
        const marker = consecutive && centre && before.centerHasFirst && !after.centerHasFirst ? ' (+ first player)' : '';
        out.push({ who, text: `took ${m.tiles} ${COLOR_NAMES[m.color!]} from ${from} → ${to}${marker}`, color: m.color! });
      }
    } else {
      const columns = m.columns ?? [];
      const placed = columns.filter((c) => c >= 0 && c < 5).length;
      const text = placed > 0 ? `placed ${placed} tile${placed === 1 ? '' : 's'} on their wall`
        : columns.includes(5) ? 'sent their completed lines to the floor' : 'scored the round';
      const delta = consecutive ? after.players[m.seat].score - before.players[m.seat].score : 0;
      out.push({ who, text: text + signed(delta) });
    }
  }
  if (after.round > before.round && next.status !== 'finished') out.push({ who: null, text: `Round ${after.round} begins` });
  return out;
}

/// Your own seat's move is news only when you did not make it by hand here:
/// a bot playing your seat, or the server playing your forced turn.
function narrated(prev: GameView, next: GameView, seat: number): boolean {
  if (seat !== next.you.seat) return true;
  return next.seats[seat]?.kind === 'bot' || prev.autoPlay;
}
