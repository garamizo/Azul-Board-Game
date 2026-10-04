import { expect, type Browser, type Page } from '@playwright/test';

export const BASE = 'http://127.0.0.1:5081';

export async function person(browser: Browser, email: string, viewport: { width: number; height: number }) {
  const context = await browser.newContext({ viewport });
  await context.addCookies([{ name: 'azul_dev_user', value: email, url: BASE }]);
  return context.newPage();
}

export async function view(page: Page, id: string) {
  return (await page.request.get(`/api/games/${id}`)).json();
}

export async function newGame(page: Page, players: number): Promise<string> {
  await page.goto('/');
  await page.getByRole('button', { name: `${players} players` }).click();
  await page.waitForURL(/\/g\/[a-z2-7]{10}$/);
  return page.url().split('/').pop()!;
}

/// Plays the viewer's turn through the UI: the first legal take that is not
/// to the floor (else any); on a wall turn, the first open target of each
/// completed line. Returns false when the turn went away meanwhile (the
/// server auto-plays forced moves after 0.2 s), so the caller just looks again.
export async function playTurn(page: Page, id: string): Promise<boolean> {
  const v = await view(page, id);
  if (!v.legal) return false;  // the turn went away (a forced move was auto-played)
  try {
    await expect(page.getByTestId('status')).toHaveAttribute('data-version', String(v.version), { timeout: 15_000 });
    const confirm = page.getByRole('button', { name: 'Confirm' });
    if (v.legal.takes) {
      const [f, c, r] = v.legal.takes.find((t: number[]) => t[2] !== 5) ?? v.legal.takes[0];
      await page.locator(`[data-factory="${f}"][data-color="${c}"]`).first().click();
      if (!(await confirm.isEnabled())) await page.locator(`[data-row="${r}"]`).first().click();
    } else {
      for (let row = 0; row < 5; row++) {
        if (v.legal.wall[row]) await page.locator(`[data-wall-target^="${row}-"]`).first().click();
      }
    }
    await confirm.click();
    await expect.poll(async () => (await view(page, id)).version, { timeout: 15_000 }).toBeGreaterThan(v.version);
    return true;
  } catch (e) {
    if ((await view(page, id)).version > v.version) return false;  // someone (the server) moved first
    throw e;
  }
}

export async function noHorizontalScroll(page: Page) {
  const [scroll, width] = await page.evaluate(() => [document.documentElement.scrollWidth, window.innerWidth]);
  expect(scroll).toBeLessThanOrEqual(width);
}
