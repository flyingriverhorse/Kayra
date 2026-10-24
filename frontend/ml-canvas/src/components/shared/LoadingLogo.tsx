import { useEffect, useState } from 'react';
import logoUrl from '../../../../../static/img/app_logo/logo.png';
import videoUrl from '../../../../../static/img/app_logo/Logo_Animasyonu.mp4';

/** Play the logo animation during loading unless reduced motion is requested. */
export function LoadingLogo() {
  const [animate, setAnimate] = useState(false);
  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setAnimate(!preference.matches);
    update();
    preference.addEventListener('change', update);
    return () => preference.removeEventListener('change', update);
  }, []);

  // The video has a light background, so it sits in a rounded tile on both themes.
  return animate ? (
    <video
      src={videoUrl}
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
