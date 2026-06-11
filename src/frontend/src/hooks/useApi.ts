/**
 * Shared API utilities and re-exports.
 *
 * apiFetch and useMutation are the building blocks used by all domain-specific hook files.
 * All hooks are re-exported here so existing imports continue to work.
 */

import { useState, useEffect, useCallback } from 'react';
import type { AppConfig } from '../types';
import { isAppConfigured } from '../utils/configUtils';
import { cachedFetch, invalidateCache } from '../utils/fetchCache';

const API_BASE = '/api';

export type ApiFetchOptions = RequestInit & { timeoutMs?: number };

function formatApiErrorDetail(detail: unknown, status: number): string {
  if (typeof detail === 'string') {
    if (
      detail.includes('Credential was not sent') ||
      detail.includes('401:') ||
      detail.includes('cannot get access token') ||
      detail.includes('token refresh') ||
      detail.includes('exit status 45')
    ) {
      return 'Databricks authentication expired. For local dev, run: databricks auth login --profile e2-demo-field-eng then restart the API server with DATABRICKS_PROFILE=e2-demo-field-eng.';
    }
    if (detail.length > 280) {
      return `${detail.slice(0, 280)}…`;
    }
    return detail;
  }
  if (Array.isArray(detail)) {
    return detail
      .map((item) =>
        typeof item === 'object' && item && 'msg' in item
          ? String((item as { msg: string }).msg)
          : String(item),
      )
      .join('; ');
  }
  if (detail && typeof detail === 'object') {
    return JSON.stringify(detail);
  }
  return `API error: ${status}`;
}

export async function apiFetch<T>(path: string, options?: ApiFetchOptions): Promise<T> {
  const { timeoutMs, ...init } = options ?? {};
  let timer: ReturnType<typeof setTimeout> | undefined;
  let signal = init.signal;
  let abortWasTimeout = false;
  if (timeoutMs != null && signal === undefined) {
    const ctrl = new AbortController();
    timer = setTimeout(() => {
      abortWasTimeout = true;
      ctrl.abort();
    }, timeoutMs);
    signal = ctrl.signal;
  }
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...init.headers },
      signal,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(formatApiErrorDetail(body.detail ?? res.statusText, res.status));
    }
    return res.json();
  } catch (e) {
    if (abortWasTimeout && e instanceof DOMException && e.name === 'AbortError') {
      throw new Error(`Request timed out after ${timeoutMs}ms`);
    }
    throw e;
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/**
 * Generic hook for POST/mutation API calls.
 * Handles loading state, error capture, and re-throw.
 */
export function useMutation<TParams, TResult>(
  fn: (params: TParams) => Promise<TResult>
) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const mutate = useCallback(
    async (params: TParams): Promise<TResult> => {
      setLoading(true);
      setError(null);
      try {
        const data = await fn(params);
        return data;
      } catch (e: any) {
        setError(e.message);
        throw e;
      } finally {
        setLoading(false);
      }
    },
    [fn]
  );

  return { mutate, loading, error };
}

// --- Config ---

export function useConfig() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [loading, setLoading] = useState(true);

  const applyConfig = useCallback((cfg: AppConfig) => {
    invalidateCache('config');
    setConfig(cfg);
  }, []);

  const refresh = useCallback(async (opts?: { force?: boolean }) => {
    if (opts?.force) invalidateCache('config');
    setLoading(true);
    try {
      const data = await cachedFetch(
        'config',
        () => apiFetch<AppConfig>('/config', { timeoutMs: 15_000 }),
        0,
      );
      setConfig(data);
    } catch {
      setConfig({
        prompt_catalog: '',
        prompt_schema: '',
        eval_catalog: '',
        eval_schema: '',
        mlflow_experiment_name: '',
        sql_warehouse_id: '',
        sql_warehouse_name: '',
        evaluate_tab_enabled: false,
        is_configured: false,
      });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const saveSettings = useCallback(async (updates: Partial<AppConfig>) => {
    invalidateCache('config');
    const saved = await apiFetch<AppConfig>('/config', {
      method: 'POST',
      body: JSON.stringify(updates),
    });
    invalidateCache('config');
    setConfig(saved);
    return saved;
  }, []);

  const isConfigured = !loading && isAppConfigured(config);

  return { config, loading, refresh, applyConfig, saveSettings, isConfigured };
}

// --- Re-exports for backward compatibility ---

export { usePrompts, usePromptVersions, usePromptTemplate, useCreatePrompt, useSaveVersion } from './usePromptApi';
export { useModels } from './useModelApi';
export { useRunPrompt } from './useRunApi';
export {
  useExperiments,
  useExperimentBrowse,
  useExperimentPrompts,
  useJudges,
  useCreateJudge,
  useDeleteJudge,
  useEvalTables,
  useEvalColumns,
  useRunEval,
} from './useEvalApi';
