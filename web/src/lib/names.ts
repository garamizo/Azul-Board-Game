import type { GameView } from './types';

const local = (email: string) => email.split('@')[0];

export function seatName(view: GameView, idx: number): string {
  const seat = view.seats[idx];
  if (view.you.seat === idx && seat.kind === 'human') return 'You';
  if (seat.kind === 'human' && seat.email) return local(seat.email);
  if (seat.kind === 'bot') return seat.email ? `${local(seat.email)} (bot)` : `Bot ${idx + 1}`;
  return 'Open seat';
}
