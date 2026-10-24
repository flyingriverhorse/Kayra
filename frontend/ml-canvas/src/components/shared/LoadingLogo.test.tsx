import { act, cleanup, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { LoadingLogo } from './LoadingLogo';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it('plays the logo video only while reduced motion is off', () => {
  // Preference changes must swap video and still logo and release the media listener on unmount.
  const preference = Object.assign(new EventTarget(), { matches: true });
  const remove = vi.spyOn(preference, 'removeEventListener');
  vi.spyOn(window, 'matchMedia').mockReturnValue(preference as MediaQueryList);
  const { container, unmount } = render(<LoadingLogo />);
  expect(container.querySelector('video')).toBeNull();
  expect(screen.getByAltText('')).toHaveAttribute('src', expect.stringContaining('logo.png'));

  act(() => {
    preference.matches = false;
    preference.dispatchEvent(new Event('change'));
  });
  const video = container.querySelector('video');
  expect(video).toHaveAttribute('src', expect.stringContaining('logo-animasyon-beyaz.mp4'));
  expect(video).toHaveAttribute('poster', expect.stringContaining('logo.png'));
  expect(video).toHaveProperty('muted', true);
  expect(video).toHaveAttribute('loop');
  expect(screen.queryByAltText('')).toBeNull();

  act(() => {
    preference.matches = true;
    preference.dispatchEvent(new Event('change'));
  });
  expect(container.querySelector('video')).toBeNull();
  expect(screen.getByAltText('')).toHaveAttribute('src', expect.stringContaining('logo.png'));
  unmount();
  expect(remove).toHaveBeenCalledWith('change', expect.any(Function));
});

it('plays the dark-background video in dark mode', () => {
  // A white video tile on a dark page would look like a broken image.
  vi.spyOn(window, 'matchMedia').mockReturnValue(Object.assign(new EventTarget(), { matches: false }) as MediaQueryList);
  document.documentElement.classList.add('dark');
  try {
    const { container } = render(<LoadingLogo />);
    expect(container.querySelector('video')).toHaveAttribute('src', expect.stringContaining('logo-animasyon-koyu.mp4'));
  } finally {
    document.documentElement.classList.remove('dark');
  }
});
