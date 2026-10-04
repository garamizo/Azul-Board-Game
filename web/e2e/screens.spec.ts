import { expect, test } from '@playwright/test';
import { newGame, person, playTurn } from './helpers';

// SCREENS=1 npx playwright test screens  →  test-results/screens/*.png for a human look.
test.skip(!process.env.SCREENS, 'screenshots only on request');

for (const scheme of ['light', 'dark'] as const) {
  for (const [label, width, height] of [['phone', 360, 740], ['desktop', 1440, 900]] as const) {
    test(`${label} ${scheme}`, async ({ browser }) => {
      const page = await person(browser, 'alice@example.com', { width, height }, { colorScheme: scheme });
      const id = await newGame(page, 4);
      await page.getByRole('button', { name: 'Start' }).click();
      await expect(page.getByTestId('status')).toBeVisible();
      // Play two rounds of turns so the opponents' cards and boards hold tiles.
      for (let i = 0; i < 2; i++) await playTurn(page, id).catch(() => false);
      await page.waitForTimeout(1500);
      await page.screenshot({ path: `test-results/screens/${label}-${scheme}.png`, fullPage: true });
    });
  }
}
