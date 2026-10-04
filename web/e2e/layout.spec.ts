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

for (const players of [2, 4]) {
  test(`desktop ${players}-player proportions: small opponents, large factories`, async ({ browser }) => {
    const alice = await person(browser, 'alice@example.com', { width: 1440, height: 900 });
    await newGame(alice, players);
    await alice.getByRole('button', { name: 'Start' }).click();
    await expect(alice.getByTestId('status')).toBeVisible();
    const mine = (await alice.locator('.mine svg.board').boundingBox())!;
    const others = alice.locator('.others-desktop svg.board');
    await expect(others).toHaveCount(players - 1);
    for (const box of await others.all()) {
      expect((await box.boundingBox())!.width).toBeLessThanOrEqual(0.6 * mine.width);
    }
    for (const factory of await alice.locator('svg.factory').all()) {
      expect((await factory.boundingBox())!.width).toBeGreaterThanOrEqual(90);
    }
    await noHorizontalScroll(alice);
  });
}
