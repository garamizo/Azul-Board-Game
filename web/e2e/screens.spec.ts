import { expect, test } from '@playwright/test';
import { newGame, person } from './helpers';

// SCREENS=1 npx playwright test screens  →  test-results/screens/*.png for a human look.
test.skip(!process.env.SCREENS, 'screenshots only on request');

for (const scheme of ['light', 'dark'] as const) {
  for (const [label, width, height] of [['phone', 360, 740], ['desktop', 1440, 900]] as const) {
    test(`${label} ${scheme}`, async ({ browser }) => {
      const page = await person(browser, 'alice@example.com', { width, height }, { colorScheme: scheme });
      await newGame(page, 4);
      await page.getByRole('button', { name: 'Start' }).click();
      await expect(page.getByTestId('status')).toBeVisible();
      await page.waitForTimeout(1500);  // let the bots move once
      await page.screenshot({ path: `test-results/screens/${label}-${scheme}.png`, fullPage: true });
    });
  }
}
