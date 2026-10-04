import { expect, test } from '@playwright/test';
import { newGame, person } from './helpers';

test('a deleted game sends watchers back to the lobby', async ({ browser }) => {
  const alice = await person(browser, 'alice@example.com', { width: 1280, height: 800 });
  const bob = await person(browser, 'bob@example.com', { width: 1280, height: 800 });
  const id = await newGame(alice, 2);
  await bob.goto(`/g/${id}`);
  await expect(bob.getByRole('button', { name: 'Take this seat' })).toBeVisible();
  alice.once('dialog', (d) => d.accept());
  await alice.getByRole('button', { name: 'Delete game' }).click();
  await expect(bob).toHaveURL(/\/$/, { timeout: 15_000 });
  await expect(bob.getByText('This game was deleted')).toBeVisible();
});
