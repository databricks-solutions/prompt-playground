import { useState, useEffect, useCallback, useRef } from 'react';
import type { PromptInfo, PromptVersion, PromptTemplate, CreatePromptResponse, SaveVersionResponse } from '../types';
import { apiFetch, useMutation } from './useApi';
import { cachedFetch, invalidateCache } from '../utils/fetchCache';

export function usePrompts(catalog: string, schema: string) {
  const [prompts, setPrompts] = useState<PromptInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!catalog || !schema) return;
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({ catalog, schema });
      const cacheKey = `prompts:${catalog}.${schema}`;
      const data = await cachedFetch(cacheKey, () =>
        apiFetch<{ prompts: PromptInfo[] }>(`/prompts?${params.toString()}`),
      );
      setPrompts(data.prompts);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [catalog, schema]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const refreshInvalidating = useCallback(async () => {
    if (catalog && schema) {
      invalidateCache(`prompts:${catalog}.${schema}`);
    }
    await refresh();
  }, [catalog, schema, refresh]);

  return { prompts, loading, error, refresh: refreshInvalidating };
}

export function usePromptVersions(name: string | null) {
  const [versions, setVersions] = useState<PromptVersion[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Tracks the most recent request so stale responses from a previous prompt are ignored
  const requestIdRef = useRef(0);

  const fetchVersions = useCallback(async () => {
    const requestId = ++requestIdRef.current;
    setVersions([]);
    if (!name) return;
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({ name });
      const data = await apiFetch<{ versions: PromptVersion[] }>(
        `/prompts/versions?${params.toString()}`
      );
      if (requestId === requestIdRef.current) setVersions(data.versions);
    } catch (e: any) {
      if (requestId === requestIdRef.current) setError(e.message);
    } finally {
      if (requestId === requestIdRef.current) setLoading(false);
    }
  }, [name]);

  useEffect(() => {
    fetchVersions();
  }, [fetchVersions]);

  const reset = useCallback(() => setVersions([]), []);

  return { versions, loading, error, refresh: fetchVersions, reset };
}

export function usePromptTemplate(name: string | null, version: string | null) {
  const [template, setTemplate] = useState<PromptTemplate | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!name || !version) {
      setTemplate(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    const params = new URLSearchParams({ name, version });
    apiFetch<PromptTemplate>(`/prompts/template?${params.toString()}`)
      .then((data) => {
        if (!cancelled) setTemplate(data);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [name, version]);

  return { template, loading, error };
}

export function useCreatePrompt() {
  const { mutate: create, loading, error } = useMutation(
    (params: { name: string; template: string; description?: string; experiment_name?: string }) =>
      apiFetch<CreatePromptResponse>('/prompts', {
        method: 'POST',
        body: JSON.stringify(params),
      })
  );
  return { create, loading, error };
}

export function useSaveVersion() {
  const { mutate: save, loading, error } = useMutation(
    (params: { name: string; template: string; description?: string }) =>
      apiFetch<SaveVersionResponse>('/prompts/versions', {
        method: 'POST',
        body: JSON.stringify(params),
      })
  );
  return { save, loading, error };
}
