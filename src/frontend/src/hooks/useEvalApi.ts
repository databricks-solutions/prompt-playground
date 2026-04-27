import { useState, useEffect, useCallback, useRef } from 'react';
import type { ExperimentInfo, JudgeInfo, EvalResponse, EvalHistoryRun, EvalTraceRow } from '../types';
import { apiFetch, useMutation } from './useApi';

export function useExperiments() {
  const [experiments, setExperiments] = useState<ExperimentInfo[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await apiFetch<{ experiments: ExperimentInfo[] }>('/eval/experiments');
      setExperiments(data.experiments);
    } catch {
      setExperiments([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { experiments, loading, refresh };
}

export function useExperimentPrompts(experimentName: string, catalog?: string, schema?: string) {
  const [promptNames, setPromptNames] = useState<string[] | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setPromptNames(null);
    if (!experimentName) return;
    setLoading(true);
    const params = new URLSearchParams({ experiment_name: experimentName });
    if (catalog) params.set('catalog', catalog);
    if (schema) params.set('schema', schema);
    try {
      const d = await apiFetch<{ prompt_names: string[] }>(`/eval/experiments/prompts?${params.toString()}`);
      setPromptNames(d.prompt_names.length > 0 ? d.prompt_names : null);
    } catch {
      setPromptNames(null);
    } finally {
      setLoading(false);
    }
  }, [experimentName, catalog, schema]);

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
    apiFetch<{ columns: string[]; rows: Record<string, string>[]; total_rows: number }>(`/eval/table-preview?${params.toString()}`)
      .then((d) => { setColumns(d.columns); setRows(d.rows); setTotalRows(d.total_rows); })
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
      const res = await fetch('/api/eval/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
        signal: abortRef.current.signal,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(body.detail || `API error: ${res.status}`);
      }
      const data: EvalResponse = await res.json();
      setResult(data);
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
