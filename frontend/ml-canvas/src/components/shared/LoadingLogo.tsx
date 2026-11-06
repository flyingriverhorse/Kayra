import { useEffect, useState } from 'react';
import logoUrl from '../../../../../static/img/app_logo/logo.png';
import darkVideoUrl from '../../../../../static/img/app_logo/logo-animasyon-koyu.mp4';
import lightVideoUrl from '../../../../../static/img/app_logo/logo-animasyon-beyaz.mp4';

/** Play the logo animation for the current theme unless reduced motion is requested. */
export function LoadingLogo() {
  const [animate, setAnimate] = useState(false);
  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setAnimate(!preference.matches);
    update();
    preference.addEventListener('change', update);
    return () => preference.removeEventListener('change', update);
  }, []);

  // Pick one source so only one video downloads; a loader rarely outlives a theme switch.
  const dark = document.documentElement.classList.contains('dark');
  // Each video has a solid background, so it sits in a rounded tile.
  return animate ? (
    <video
      src={dark ? darkVideoUrl : lightVideoUrl}
      poster={logoUrl}
      width={64}
      height={64}
      className="h-16 w-16 rounded-xl object-cover"
      autoPlay
      muted
      loop
      playsInline
      aria-hidden="true"
    />
  ) : (
    <img src={logoUrl} alt="" width={64} height={64} className="h-16 w-16 object-contain" aria-hidden="true" />
  );
}
