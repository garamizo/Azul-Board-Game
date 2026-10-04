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
  const card = alice.locator('.others-phone .card').first();
  await expect(card.locator('.line i')).toHaveCount(15);
  await expect(card.locator('.wall i')).toHaveCount(25);
  await expect(card.locator('.floor i')).toHaveCount(7);
  await expect(alice.locator('svg.factory')).toHaveCount(9);
});

for (const width of [900, 1120, 1440]) {
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

test('dark mode follows the system setting', async ({ browser }) => {
  const page = await person(browser, 'alice@example.com', { width: 900, height: 900 }, { colorScheme: 'dark' });
  await page.goto('/');
  await expect(page.getByRole('button', { name: '2 players' })).toBeVisible();
  expect(await page.evaluate(() => getComputedStyle(document.body).backgroundColor)).toBe('rgb(15, 26, 43)');
});

test('1440 px, 4 players: factories on a ring around a round centre', async ({ browser }) => {
  const alice = await person(browser, 'alice@example.com', { width: 1440, height: 900 });
  await newGame(alice, 4);
  await alice.getByRole('button', { name: 'Start' }).click();
  await expect(alice.getByTestId('status')).toBeVisible();
  const boxes = await Promise.all((await alice.locator('svg.factory').all()).map(async (f) => (await f.boundingBox())!));
  expect(boxes).toHaveLength(9);
  const c = boxes.map((b) => ({ x: b.x + b.width / 2, y: b.y + b.height / 2, d: b.width }));
  for (let i = 0; i < c.length; i++) {
    const n = c[(i + 1) % c.length];
    expect(c[i].d).toBeGreaterThanOrEqual(90);
    expect(Math.hypot(n.x - c[i].x, n.y - c[i].y)).toBeGreaterThanOrEqual(c[i].d);  // circles apart
  }
  const ring = (await alice.locator('.tray .factories').boundingBox())!;
  const centre = (await alice.locator('[data-flight-source="9"]').boundingBox())!;
  expect(Math.abs(centre.x + centre.width / 2 - (ring.x + ring.width / 2))).toBeLessThan(2);
  expect(Math.abs(centre.y + centre.height / 2 - (ring.y + ring.height / 2))).toBeLessThan(2);
  await noHorizontalScroll(alice);
});

test('480 px phone keeps the factory grid', async ({ browser }) => {
  const alice = await person(browser, 'alice@example.com', { width: 480, height: 900 });
  await newGame(alice, 4);
  await alice.getByRole('button', { name: 'Start' }).click();
  await expect(alice.getByTestId('status')).toBeVisible();
  expect(await alice.locator('.tray .slot').first().evaluate((el) => getComputedStyle(el).position)).toBe('static');
  await noHorizontalScroll(alice);
});
