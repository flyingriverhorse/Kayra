import { useQuery } from '@tanstack/react-query';
import { fetchUploadConfig, type UploadConfig } from '../api/client';

/**
 * Server-side upload limits (max size + allowed extensions), fetched once
 * per session from `GET /api/config`. Callers keep hardcoded fallbacks for
 * the brief window before the first response (and for offline/dev use), so
 * this hook's absence of data must never block the upload UI.
 */
export const useUploadConfig = () =>
  useQuery<UploadConfig>({
    queryKey: ['upload-config'],
    queryFn: fetchUploadConfig,
    staleTime: 5 * 60 * 1000,
    retry: 1,
  });

/**
 * True when the server runs the public demo (`DEMO_MODE`): only the bundled
 * Iris dataset exists, and upload, new-source and delete controls are hidden
 * because the server refuses them. False until the config has loaded.
 */
export const useDemoMode = (): boolean => useUploadConfig().data?.demo_mode ?? false;
