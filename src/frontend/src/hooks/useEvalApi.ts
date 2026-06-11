import { useState, useEffect, useCallback, useRef } from 'react';
import type { ExperimentInfo, JudgeInfo, EvalResponse, EvalHistoryRun, EvalTraceRow } from '../types';
import { apiFetch, useMutation } from './useApi';
import { cachedFetch } from '../utils/fetchCache';

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

interface EvalJobStatus {
  job_id: string;
  status: string;
  progress?: number;
  total?: number;
  message?: string;
  result?: EvalResponse;
  error?: string | null;
}

function mergeExperiments(
  ...lists: Array<ExperimentInfo[] | undefined>
): ExperimentInfo[] {
  const byName = new Map<string, ExperimentInfo>();
  for (const list of lists) {
    for (const exp of list ?? []) {
      if (exp?.name) byName.set(exp.name, exp);
    }
  }
  return [...byName.values()].sort((a, b) => a.name.localeCompare(b.name));
}

export function useExperimentBrowse(
  enabled: boolean,
  catalog?: string,
  schema?: string,
) {
  const [experiments, setExperiments] = useState<ExperimentInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const searchTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const browseExperiments = useCallback(
    async (search?: string) => {
      if (!enabled) return [];
      setLoading(true);
      setError(null);
      const query = search?.trim() ?? '';
      const params = new URLSearchParams({ browse: 'true' });
      const cat = catalog?.trim() ?? '';
      const sch = schema?.trim() ?? '';
      if (query.length >= 2) {
        params.set('q', query);
      } else if (cat && sch) {
        params.set('catalog', cat);
        params.set('schema', sch);
      }
      const cacheKey = `experiments:browse:${cat}:${sch}:${query}`;
      try {
        const data = await cachedFetch(cacheKey, () =>
          apiFetch<{ experiments: ExperimentInfo[] }>(
            `/eval/experiments?${params.toString()}`,
            { timeoutMs: 90_000 },
          ),
        );
        setExperiments((prev) => mergeExperiments(prev, data.experiments));
        return data.experiments;
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Could not load MLflow experiments');
        return [];
      } finally {
        setLoading(false);
      }
    },
    [enabled, catalog, schema],
  );

  const handleOpen = useCallback(() => {
    apiFetch('/eval/experiments/warm', { method: 'POST' }).catch(() => {});
    void browseExperiments();
  }, [browseExperiments]);

  const handleQueryChange = useCallback(
    (query: string) => {
      if (searchTimer.current) clearTimeout(searchTimer.current);
      const trimmed = query.trim();
      if (trimmed.length < 2) {
        void browseExperiments();
        return;
      }
      searchTimer.current = setTimeout(() => {
        void browseExperiments(trimmed);
      }, 400);
    },
    [browseExperiments],
  );

  useEffect(() => {
    if (!enabled) {
      setExperiments([]);
      setLoading(false);
      setError(null);
      return;
    }
    setError(null);
    apiFetch('/eval/experiments/warm', { method: 'POST' }).catch(() => {});
    return () => {
      if (searchTimer.current) clearTimeout(searchTimer.current);
    };
  }, [enabled]);

  return {
    experiments,
    setExperiments,
    loading,
    error,
    browseExperiments,
    onOpen: handleOpen,
    onQueryChange: handleQueryChange,
  };
}

export function useExperiments(enabled: boolean, catalog?: string, schema?: string) {
  const browse = useExperimentBrowse(enabled, catalog, schema);

  useEffect(() => {
    if (!enabled) return;
    void browse.browseExperiments();
  }, [enabled, catalog, schema, browse.browseExperiments]);

  return {
    experiments: browse.experiments,
    loading: browse.loading,
    error: browse.error,
    refresh: browse.browseExperiments,
    onExperimentOpen: browse.onOpen,
    onExperimentQueryChange: browse.onQueryChange,
  };
}

export function useExperimentPrompts(
  experimentName: string,
  catalog?: string,
  schema?: string,
  enabled: boolean = true,
) {
  const [promptNames, setPromptNames] = useState<string[] | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    if (!enabled) {
      setPromptNames(null);
      setLoading(false);
      return;
    }
    setPromptNames(null);
    if (!experimentName) return;
    setLoading(true);
    const params = new URLSearchParams({ experiment_name: experimentName });
    if (catalog) params.set('catalog', catalog);
    if (schema) params.set('schema', schema);
    try {
      const d = await apiFetch<{ prompt_names: string[] }>(
        `/eval/experiments/prompts?${params.toString()}`,
        { timeoutMs: 25_000 },
      );
      setPromptNames(d.prompt_names.length > 0 ? d.prompt_names : null);
    } catch {
      setPromptNames(null);
    } finally {
      setLoading(false);
    }
  }, [experimentName, catalog, schema, enabled]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { promptNames, loading, refresh };
}

export function useJudges(experimentName: string) {
  const [judges, setJudges] = useState<JudgeInfo[]>([]);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(() => {
    if (!experimentName) {
      setJudges([]);
      return;
    }
    setLoading(true);
    const params = new URLSearchParams({ experiment_name: experimentName });
    apiFetch<{ judges: JudgeInfo[] }>(`/eval/judges?${params.toString()}`)
      .then((d) => setJudges(d.judges))
      .catch(() => setJudges([]))
      .finally(() => setLoading(false));
  }, [experimentName]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { judges, loading, refresh };
}

export function useCreateJudge() {
  const { mutate: create, loading, error } = useMutation(
    (params: {
      name: string;
      type?: 'custom' | 'guidelines';
      instructions?: string;
      guidelines?: string[];
      experiment_name?: string;
      is_update?: boolean;
    }) =>
      apiFetch<{ name: string; status: string }>('/eval/judges', {
        method: 'POST',
        body: JSON.stringify(params),
      })
  );
  return { create, loading, error };
}

export interface JudgeDetail {
  name: string;
  type: 'custom' | 'guidelines';
  instructions: string | null;
  guidelines: string[] | null;
}

export function useJudgeDetail(name: string | null) {
  const [detail, setDetail] = useState<JudgeDetail | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!name) { setDetail(null); return; }
    setLoading(true);
    const params = new URLSearchParams({ name });
    apiFetch<JudgeDetail>(`/eval/judges/detail?${params.toString()}`)
      .then((d) => setDetail(d))
      .catch(() => setDetail(null))
      .finally(() => setLoading(false));
  }, [name]);

  return { detail, loading };
}

export function useDeleteJudge() {
  const { mutate: deleteJudge, loading, error } = useMutation(
    (params: { name: string; experiment_name?: string }) => {
      const qs = new URLSearchParams({ name: params.name });
      if (params.experiment_name) qs.set('experiment_name', params.experiment_name);
      return apiFetch<{ name: string; status: string }>(`/eval/judges?${qs.toString()}`, {
        method: 'DELETE',
      });
    }
  );
  return { deleteJudge, loading, error };
}

export function useEvalTables(catalog: string, schema: string) {
  const [tables, setTables] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!catalog || !schema) return;
    setLoading(true);
    setError(null);
    const params = new URLSearchParams({ catalog, schema });
    apiFetch<{ tables: { name: string }[] }>(`/eval/tables?${params.toString()}`)
      .then((d) => setTables(d.tables.map((t) => t.name)))
      .catch((e) => { setTables([]); setError(String(e)); })
      .finally(() => setLoading(false));
  }, [catalog, schema]);

  return { tables, loading, error };
}

export function useEvalColumns(catalog: string, schema: string, table: string | null) {
  const [columns, setColumns] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!table) { setColumns([]); return; }
    setLoading(true);
    const params = new URLSearchParams({ catalog, schema, table });
    apiFetch<{ columns: string[] }>(`/eval/columns?${params.toString()}`)
      .then((d) => setColumns(d.columns))
      .catch(() => setColumns([]))
      .finally(() => setLoading(false));
  }, [catalog, schema, table]);

  return { columns, loading };
}

export function useTablePreview(catalog: string, schema: string, table: string | null, limit = 20) {
  const [columns, setColumns] = useState<string[]>([]);
  const [rows, setRows] = useState<Record<string, string>[]>([]);
  const [totalRows, setTotalRows] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!table) { setColumns([]); setRows([]); setTotalRows(null); return; }
    setLoading(true);
    const params = new URLSearchParams({ catalog, schema, table, limit: String(limit) });
    apiFetch<{ columns: string[]; rows: Record<string, string>[]; total_rows: number | null }>(`/eval/table-preview?${params.toString()}`)
      .then((d) => { setColumns(d.columns); setRows(d.rows); setTotalRows(d.total_rows ?? null); })
      .catch(() => { setColumns([]); setRows([]); setTotalRows(null); })
      .finally(() => setLoading(false));
  }, [catalog, schema, table]);

  return { columns, rows, totalRows, loading };
}

export function useRunEval() {
  const [result, setResult] = useState<EvalResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const runEval = useCallback(async (params: {
    prompt_name: string;
    prompt_version: string;
    model_name: string;
    dataset_catalog: string;
    dataset_schema: string;
    dataset_table: string;
    column_mapping: Record<string, string>;
    max_rows?: number;
    temperature?: number;
    experiment_name?: string;
    scorer_name?: string;
    judge_model?: string;
    judge_temperature?: number;
    expectations_column?: string;
  }) => {
    abortRef.current?.abort();
    abortRef.current = new AbortController();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const start = await apiFetch<{ job_id: string; status: string }>('/eval/run', {
        method: 'POST',
        body: JSON.stringify(params),
        signal: abortRef.current.signal,
        timeoutMs: 60_000,
      });
      let status = start.status;
      let jobId = start.job_id;
      while (status === 'pending' || status === 'running') {
        if (abortRef.current?.signal.aborted) {
          throw new DOMException('Aborted', 'AbortError');
        }
        await sleep(1500);
        const poll = await apiFetch<EvalJobStatus>(
          `/eval/run/status?${new URLSearchParams({ job_id: jobId }).toString()}`,
          { signal: abortRef.current.signal, timeoutMs: 30_000 },
        );
        status = poll.status;
        if (status === 'failed') {
          throw new Error(poll.error || 'Evaluation failed');
        }
        if (status === 'completed' && poll.result) {
          setResult(poll.result);
          return;
        }
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') setError(e.message);
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }, []);

  const abort = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { result, loading, error, runEval, abort, reset };
}

export function useRunTraces(runId: string | null) {
  const [rows, setRows] = useState<EvalTraceRow[]>([]);
  const [scorer, setScorer] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId) { setRows([]); setScorer(''); return; }
    setLoading(true);
    setError(null);
    const params = new URLSearchParams({ run_id: runId });
    apiFetch<{ scorer: string; rows: EvalTraceRow[] }>(`/eval/run-traces?${params}`)
      .then((d) => { setRows(d.rows); setScorer(d.scorer); })
      .catch((e) => { setRows([]); setError(String(e)); })
      .finally(() => setLoading(false));
  }, [runId]);

  return { rows, scorer, loading, error };
}

export function useEvalHistory(promptName: string | null, experimentName: string) {
  const [runs, setRuns] = useState<EvalHistoryRun[]>([]);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(() => {
    if (!promptName) {
      setRuns([]);
      return;
    }
    setLoading(true);
    const params = new URLSearchParams({ prompt_name: promptName });
    if (experimentName) params.set('experiment_name', experimentName);
    apiFetch<{ runs: EvalHistoryRun[] }>(`/eval/history?${params}`)
      .then((d) => setRuns(d.runs))
      .catch(() => setRuns([]))
      .finally(() => setLoading(false));
  }, [promptName, experimentName]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { runs, loading, refresh };
}
