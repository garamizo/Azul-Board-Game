function parse(path: string): string | null {
  const m = /^\/g\/([a-z2-7]{10})$/.exec(path);
  return m ? m[1] : null;
}

export const route = $state({ gameId: parse(location.pathname) });
/// One-shot message for the next page (e.g. "This game was deleted").
export const flash = $state({ text: '' });

export function navigate(path: string, message = ''): void {
  flash.text = message;
  history.pushState({}, '', path);
  route.gameId = parse(location.pathname);
}

window.addEventListener('popstate', () => { route.gameId = parse(location.pathname); });
