/// True when animations should be skipped: the user asked the system for less
/// motion, or the browser (or jsdom) has no Web Animations API.
export function reducedMotion(): boolean {
  if (typeof Element === 'undefined' || typeof Element.prototype.animate !== 'function') return true;
  return typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
}
