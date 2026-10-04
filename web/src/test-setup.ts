// jsdom has no matchMedia. svelte/motion calls it when imported, and
// reducedMotion() asks it about prefers-reduced-motion. Nothing matches by
// default; a test that wants reduced motion spies on window.matchMedia.
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  configurable: true,
  value: (query: string) => ({
    matches: false, media: query, onchange: null,
    addEventListener: () => {}, removeEventListener: () => {}, addListener: () => {}, removeListener: () => {},
    dispatchEvent: () => false,
  }),
});
