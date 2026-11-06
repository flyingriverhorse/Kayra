import { test, expect } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

for (const width of [1440, 390]) {
  test(`application branding and loading motion at ${width}px`, async ({ page }, testInfo) => {
    // The loader plays the light-theme logo video; reduced motion shows the still logo and fetches no video.
    await mockBackend(page);
    await page.setViewportSize({ width, height: 900 });
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await page.addInitScript(() => localStorage.setItem('skyulf-theme', 'light'));
    const mediaRequests: string[] = [];
    page.on('request', request => { if (/\.(mp4|webm)(\?|$)/.test(request.url())) mediaRequests.push(request.url()); });
    let finishLoading!: () => void;
    const loading = new Promise<void>(resolve => { finishLoading = resolve; });
    await page.route('**/data/api/sources', async route => {
      await loading;
      await route.fulfill({ json: { sources: [] } });
    });
    await page.goto('/data');
    await expect(page).toHaveTitle('Kayra');
    const status = page.getByRole('status').filter({ hasText: 'Loading datasets...' });
    const video = status.locator('video');
    await expect(video).toBeVisible();
    await expect(video).toHaveAttribute('src', /logo-animasyon-beyaz.*\.mp4/);
    await expect(video).toHaveAttribute('loop', '');
    expect(await video.evaluate(element => (element as HTMLVideoElement).muted)).toBe(true);
    await page.screenshot({ path: testInfo.outputPath('loading-light.png') });

    const favicon = page.locator('link[rel="icon"]');
    await expect(favicon).toHaveAttribute('type', 'image/png');
    const faviconUrl = await favicon.evaluate(element => (element as HTMLLinkElement).href);
    const response = await page.request.get(faviconUrl);
    expect(response.ok()).toBe(true);
    expect((await response.body()).subarray(0, 8).toString('hex')).toBe('89504e470d0a1a0a');

    if (width < 768) await page.getByRole('button', { name: 'Open navigation menu' }).click();
    const sidebar = page.locator('#app-sidebar');
    const logo = sidebar.getByRole('img', { name: 'Kayra logo', exact: true });
    const byline = sidebar.getByText('by Skyulf', { exact: true });
    await expect(byline).toBeVisible();
    await expect(byline.locator('..')).toHaveText('Kayra by Skyulf');
    await expect(logo).toBeVisible();
    await expect.poll(() => logo.evaluate(element => (element as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
    await sidebar.getByRole('button', { name: 'Switch to dark mode' }).click();
    await expect(page.locator('html')).toHaveClass(/dark/);
    await page.screenshot({ path: testInfo.outputPath('navigation-dark.png') });
    if (width < 768) await page.getByRole('button', { name: 'Close navigation menu' }).click();

    mediaRequests.length = 0;
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await expect(status.locator('video')).toHaveCount(0);
    await expect(status.locator('img')).toBeVisible();
    expect(mediaRequests).toEqual([]);
    finishLoading();
    await expect(status).toHaveCount(0);
  });
}
