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
