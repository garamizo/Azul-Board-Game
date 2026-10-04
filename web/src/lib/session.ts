const KEY = 'azul.reloadedForAuth';

/// An expired Access session answers 401 (we send X-Requested-With). Reload
/// once to go through the sign-in page; a second 401 before any success
/// means reloading will not help.
export function handleUnauthorized(win: Window = window): 'reloading' | 'signed-out' {
  try {
    if (win.sessionStorage.getItem(KEY)) return 'signed-out';
    win.sessionStorage.setItem(KEY, String(Date.now()));
  } catch {
    return 'signed-out';
  }
  win.location.reload();
  return 'reloading';
}

export function clearReloadStamp(win: Window = window): void {
  try { win.sessionStorage.removeItem(KEY); } catch { /* storage blocked */ }
}
