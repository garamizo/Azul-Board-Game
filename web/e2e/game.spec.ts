import { execSync } from 'node:child_process';
import { expect, test } from '@playwright/test';
import { newGame, noHorizontalScroll, person, playTurn, view } from './helpers';

test('two people and a bot finish a 3-player game, surviving a server restart', async ({ browser }) => {
  const alice = await person(browser, 'alice@example.com', { width: 360, height: 740 });   // phone
  const bob = await person(browser, 'bob@example.com', { width: 1280, height: 800 });      // desktop
  const id = await newGame(alice, 3);
  await bob.goto(`/g/${id}`);
  await bob.getByRole('button', { name: 'Take this seat' }).first().click();
  await alice.getByRole('button', { name: 'Start' }).click();
  await expect(alice.getByTestId('status')).toBeVisible();
  await expect(bob.getByTestId('status')).toBeVisible();

  let moves = 0;
  let restarted = false;
  for (let i = 0; i < 1000; i++) {
    const v = await view(alice, id);
    if (v.status === 'finished') break;
    if (!restarted && moves >= 6) {
      execSync('make -C .. e2e-server-restart', { stdio: 'inherit' });
      restarted = true;
      await expect(alice.getByText('Reconnecting…')).toBeHidden({ timeout: 30_000 });
      await expect(bob.getByText('Reconnecting…')).toBeHidden({ timeout: 30_000 });
    }
    const who = v.legal ? alice : (await view(bob, id)).legal ? bob : null;
    if (who) {
      if (await playTurn(who, id)) moves++;
    } else {
      await alice.waitForTimeout(150);
    }
  }
  expect(restarted).toBe(true);
  await expect(alice.getByTestId('result')).toBeVisible({ timeout: 15_000 });
  await expect(bob.getByTestId('result')).toBeVisible({ timeout: 15_000 });
  await noHorizontalScroll(alice);
});

test('on a phone a bot move shows a bubble that taps go through, except its ×', async ({ browser }) => {
  const alice = await person(browser, 'alice@example.com', { width: 360, height: 740 }, { reducedMotion: 'no-preference' });
  const id = await newGame(alice, 2);
  await alice.getByRole('button', { name: 'Start' }).click();
  await expect(alice.getByTestId('status')).toBeVisible();
  // Bubbles last 3 s, so record each as it appears instead of racing to catch it:
  // its text, whether a tap at its middle reaches the page under it (hit testing
  // honours pointer-events), and whether a tap on × reaches the × button.
  await alice.evaluate(() => {
    const seen: { text: string; through: boolean; close: boolean; dot: string | null }[] = [];
    (window as unknown as { bubbles: typeof seen }).bubbles = seen;
    const done = new WeakSet<Element>();
    new MutationObserver(() => {
      for (const b of document.querySelectorAll('[data-testid="toast"]')) {
        if (done.has(b)) continue;
        done.add(b);
        const r = b.getBoundingClientRect();
        const under = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
        const x = b.querySelector('button')!.getBoundingClientRect();
        const hit = document.elementFromPoint(x.left + x.width / 2, x.top + x.height / 2);
        const dot = b.querySelector('.dot');
        seen.push({ text: b.textContent ?? '', through: !under?.closest('[data-testid="toast"]'),
          close: !!hit?.closest('button[aria-label="Dismiss"]'), dot: dot && getComputedStyle(dot).boxShadow });
      }
    }).observe(document.body, { childList: true, subtree: true });
  });
  const moves = async () => (await alice.evaluate(() => (window as unknown as { bubbles: { text: string; dot: string | null }[] }).bubbles))
    .filter((b) => /took|placed|sent|scored/.test(b.text));
  for (let i = 0; i < 40 && (await moves()).length === 0; i++) {
    if (!(await playTurn(alice, id))) await alice.waitForTimeout(200);
  }
  const [first] = await moves();
  expect(first).toBeDefined();
  expect(first.text).toContain('Bot 2');
  if (/took \d/.test(first.text)) expect(first.dot).toContain('inset');  // the colour dot has an edge in any theme
  const all = await alice.evaluate(() => (window as unknown as { bubbles: { through: boolean; close: boolean }[] }).bubbles);
  expect(all.every((b) => b.through && b.close)).toBe(true);
});
