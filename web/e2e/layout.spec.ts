import { expect, test } from '@playwright/test';
import { newGame, noHorizontalScroll, person } from './helpers';

test('4 players on a 360 px phone with a very long email', async ({ browser }) => {
  const long = 'someone.with.an.extraordinarily.long.address.for.testing@example-with-a-long-domain.com';
  const alice = await person(browser, 'alice@example.com', { width: 360, height: 740 });
  const other = await person(browser, long, { width: 360, height: 740 });
  const id = await newGame(alice, 4);
  await other.goto(`/g/${id}`);
  await other.getByRole('button', { name: 'Take this seat' }).first().click();
  await noHorizontalScroll(alice);
  await alice.getByRole('button', { name: 'Start' }).click();
  await expect(alice.getByTestId('status')).toBeVisible();
  await expect(other.getByTestId('status')).toBeVisible();
  await noHorizontalScroll(alice);
  await noHorizontalScroll(other);
  await expect(alice.locator('svg.factory')).toHaveCount(9);
});

for (const width of [900, 1440]) {
  for (const players of [2, 3, 4]) {
    test(`desktop ${width} px, ${players} players: small opponents, large factories`, async ({ browser }) => {
      const alice = await person(browser, 'alice@example.com', { width, height: 900 });
      await newGame(alice, players);
      await alice.getByRole('button', { name: 'Start' }).click();
      await expect(alice.getByTestId('status')).toBeVisible();
      const mine = (await alice.locator('.mine svg.board').boundingBox())!;
      const others = alice.locator('.others-desktop svg.board');
      await expect(others).toHaveCount(players - 1);
      // Same size at any player count: each opponent is one of three slots.
      const widths = await Promise.all((await others.all()).map(async (b) => (await b.boundingBox())!.width));
      for (const w of widths) {
        expect(w).toBeLessThanOrEqual(0.6 * mine.width);
        expect(Math.abs(w - widths[0])).toBeLessThan(1);
      }
      for (const factory of await alice.locator('svg.factory').all()) {
        expect((await factory.boundingBox())!.width).toBeGreaterThanOrEqual(90);
      }
      await noHorizontalScroll(alice);
    });
  }
}
