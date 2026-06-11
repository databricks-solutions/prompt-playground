import { useState, useCallback, useRef } from 'react';
import type { ModelEndpoint } from '../types';
import { apiFetch } from './useApi';
import { cachedFetch } from '../utils/fetchCache';

export function useModels() {
  const [models, setModels] = useState<ModelEndpoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fetchedRef = useRef(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await cachedFetch('models', () =>
        apiFetch<{ models: ModelEndpoint[] }>('/models'),
      );
      setModels(data.models);
      fetchedRef.current = true;
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const ensureLoaded = useCallback(() => {
    if (!fetchedRef.current) refresh();
  }, [refresh]);

  return { models, loading, error, refresh, ensureLoaded };
}
