import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { Info, Loader2, Save, X } from 'lucide-react';
import SearchableSelect from './SearchableSelect';
import type { AppConfig } from '../types';
import { apiFetch, useExperimentBrowse } from '../hooks/useApi';
import { invalidateCache } from '../utils/fetchCache';
import { experimentSelectOptions } from '../utils/experimentUtils';

interface Props {
  config: AppConfig;
  onSave: (updated: AppConfig) => void;
  onClose: () => void;
}

interface Warehouse {
  id: string;
  name: string;
}

function mergeSelectOptions(values: string[], ...seeds: (string | undefined)[]) {
  const names = new Set(values);
  for (const seed of seeds) {
    const trimmed = seed?.trim();
    if (trimmed) names.add(trimmed);
  }
  return [...names].sort().map((value) => ({ value, label: value }));
}

function configuredCatalogSeeds(cfg: AppConfig): string[] {
  const names = new Set<string>();
  for (const c of [cfg.prompt_catalog, cfg.eval_catalog]) {
    const trimmed = c?.trim();
    if (trimmed) names.add(trimmed);
  }
  return [...names].sort();
}

export default function SettingsModal({ config, onSave, onClose }: Props) {
  // Form state — pre-populated from current config
  const [catalog, setCatalog] = useState(config.prompt_catalog);
  const [promptSchema, setPromptSchema] = useState(config.prompt_schema);
  const [evalCatalog, setEvalCatalog] = useState(config.eval_catalog || config.prompt_catalog);
  const [evalSchema, setEvalSchema] = useState(config.eval_schema);
  const [warehouseId, setWarehouseId] = useState(config.sql_warehouse_id);
  const [warehouseName, setWarehouseName] = useState(config.sql_warehouse_name);
  const [evaluateTabEnabled, setEvaluateTabEnabled] = useState(!!config.evaluate_tab_enabled);
  const [mlflowExperiment, setMlflowExperiment] = useState(config.mlflow_experiment_name || '');

  // Discovery state
  const [catalogs, setCatalogs] = useState<string[]>(() => configuredCatalogSeeds(config));
  const [promptSchemas, setPromptSchemas] = useState<string[]>([]);
  const [evalSchemas, setEvalSchemas] = useState<string[]>([]);
  const [warehouses, setWarehouses] = useState<Warehouse[]>([]);
  const [catalogsLoading, setCatalogsLoading] = useState(false);
  const [promptSchemasLoading, setPromptSchemasLoading] = useState(false);
  const [evalSchemasLoading, setEvalSchemasLoading] = useState(false);
  const [warehousesLoading, setWarehousesLoading] = useState(false);

  const [catalogsError, setCatalogsError] = useState<string | null>(null);
  const [warehousesError, setWarehousesError] = useState<string | null>(null);
  const [promptSchemasError, setPromptSchemasError] = useState<string | null>(null);
  const [evalSchemasError, setEvalSchemasError] = useState<string | null>(null);

  // Save state
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /** Separate in-flight flags so configured load does not block search (and vice versa) */
  const configuredCatalogsInFlight = useRef(false);
  const catalogSearchInFlight = useRef(false);
  const warehousesInFlight = useRef(false);
  const catalogSearchTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const promptSchemaSearchTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const evalSchemaSearchTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const promptSchemasInFlight = useRef(false);
  const evalSchemasInFlight = useRef(false);

  // Search results only — current selection shows in the closed trigger via `value`.
  const catalogOptions = useMemo(
    () => mergeSelectOptions(catalogs, catalog, evalCatalog),
    [catalogs, catalog, evalCatalog],
  );
  const promptSchemaOptions = useMemo(
    () => mergeSelectOptions(promptSchemas, promptSchema),
    [promptSchemas, promptSchema],
  );
  const evalSchemaOptions = useMemo(
    () => mergeSelectOptions(evalSchemas, evalSchema),
    [evalSchemas, evalSchema],
  );

  const {
    experiments: settingsExperiments,
    loading: settingsExperimentsLoading,
    error: settingsExperimentsError,
    onOpen: onSettingsExperimentOpen,
    onQueryChange: onSettingsExperimentQueryChange,
    browseExperiments: browseSettingsExperiments,
  } = useExperimentBrowse(true, catalog, promptSchema);

  const settingsExperimentOptions = useMemo(() => {
    const merged = experimentSelectOptions(settingsExperiments);
    const current = mlflowExperiment.trim();
    if (current && !merged.some((o) => o.value === current)) {
      merged.unshift(...experimentSelectOptions([{ name: current }]));
    }
    return merged;
  }, [settingsExperiments, mlflowExperiment]);

  const seedConfiguredSchema = useCallback(
    (schema: string, setter: React.Dispatch<React.SetStateAction<string[]>>) => {
      const trimmed = schema?.trim();
      if (!trimmed) return;
      setter((prev) => [...new Set([...prev, trimmed])].sort());
    },
    [],
  );

  const loadConfiguredCatalogs = useCallback((opts?: { silent?: boolean }) => {
    if (configuredCatalogsInFlight.current) return;
    configuredCatalogsInFlight.current = true;
    if (!opts?.silent) {
      setCatalogsError(null);
    }
    apiFetch<{ catalogs: string[] }>('/setup/catalogs?configured_only=true')
      .then((d) => {
        setCatalogs((prev) => {
          const merged = new Set([...prev, ...d.catalogs]);
          return [...merged].sort();
        });
        setCatalogsError(null);
      })
      .catch((e: unknown) => {
        if (!opts?.silent) {
          setCatalogsError(e instanceof Error ? e.message : 'Could not load catalogs');
        }
      })
      .finally(() => {
        configuredCatalogsInFlight.current = false;
      });
  }, []);

  const searchCatalogs = useCallback((search: string) => {
    const query = search.trim();
    if (query.length < 2) return;
    if (catalogSearchInFlight.current) return;
    catalogSearchInFlight.current = true;
    setCatalogsLoading(true);
    setCatalogsError(null);
    apiFetch<{ catalogs: string[] }>(
      `/setup/catalogs?${new URLSearchParams({ q: query }).toString()}`,
    )
      .then((d) => {
        setCatalogs((prev) => {
          const merged = new Set([...d.catalogs, catalog, evalCatalog, ...prev]);
          return [...merged].filter(Boolean).sort();
        });
        setCatalogsError(null);
      })
      .catch((e: unknown) => {
        setCatalogsError(e instanceof Error ? e.message : 'Could not search catalogs');
      })
      .finally(() => {
        catalogSearchInFlight.current = false;
        setCatalogsLoading(false);
      });
  }, [catalog, evalCatalog]);

  const handleCatalogOpen = useCallback(() => {
    loadConfiguredCatalogs();
  }, [loadConfiguredCatalogs]);

  const handleCatalogQueryChange = useCallback(
    (query: string) => {
      if (catalogSearchTimer.current) clearTimeout(catalogSearchTimer.current);
      const trimmed = query.trim();
      if (trimmed.length < 2) {
        loadConfiguredCatalogs();
        return;
      }
      catalogSearchTimer.current = setTimeout(() => {
        searchCatalogs(trimmed);
      }, 400);
    },
    [loadConfiguredCatalogs, searchCatalogs],
  );

  const loadConfiguredPromptSchemas = useCallback((cat: string) => {
    if (!cat) return;
    setPromptSchemasLoading(true);
    setPromptSchemasError(null);
    const params = new URLSearchParams({ catalog: cat, configured_only: 'true' });
    apiFetch<{ schemas: string[] }>(`/setup/schemas?${params}`)
      .then((d) => {
        setPromptSchemas(d.schemas);
        setPromptSchemasError(null);
      })
      .catch((e: unknown) => {
        setPromptSchemasError(e instanceof Error ? e.message : 'Could not load schemas');
      })
      .finally(() => setPromptSchemasLoading(false));
  }, []);

  const searchPromptSchemas = useCallback((cat: string, query: string) => {
    if (!cat || query.trim().length < 2) return;
    setPromptSchemasLoading(true);
    setPromptSchemasError(null);
    const params = new URLSearchParams({ catalog: cat, q: query.trim() });
    apiFetch<{ schemas: string[] }>(`/setup/schemas?${params}`)
      .then((d) => {
        setPromptSchemas(d.schemas);
        setPromptSchemasError(null);
      })
      .catch((e: unknown) => {
        setPromptSchemasError(e instanceof Error ? e.message : 'Could not search schemas');
      })
      .finally(() => setPromptSchemasLoading(false));
  }, []);

  const loadConfiguredEvalSchemas = useCallback((cat: string) => {
    if (!cat) return;
    setEvalSchemasLoading(true);
    setEvalSchemasError(null);
    const params = new URLSearchParams({ catalog: cat, configured_only: 'true' });
    apiFetch<{ schemas: string[] }>(`/setup/schemas?${params}`)
      .then((d) => {
        setEvalSchemas(d.schemas);
        setEvalSchemasError(null);
      })
      .catch((e: unknown) => {
        setEvalSchemasError(e instanceof Error ? e.message : 'Could not load schemas');
      })
      .finally(() => setEvalSchemasLoading(false));
  }, []);

  const searchEvalSchemas = useCallback((cat: string, query: string) => {
    if (!cat || query.trim().length < 2) return;
    setEvalSchemasLoading(true);
    setEvalSchemasError(null);
    const params = new URLSearchParams({ catalog: cat, q: query.trim() });
    apiFetch<{ schemas: string[] }>(`/setup/schemas?${params}`)
      .then((d) => {
        setEvalSchemas(d.schemas);
        setEvalSchemasError(null);
      })
      .catch((e: unknown) => {
        setEvalSchemasError(e instanceof Error ? e.message : 'Could not search schemas');
      })
      .finally(() => setEvalSchemasLoading(false));
  }, []);

  const loadWarehouses = () => {
    if (warehousesInFlight.current) return;
    warehousesInFlight.current = true;
    setWarehousesLoading(true);
    setWarehousesError(null);
    apiFetch<{ warehouses: Warehouse[] }>('/setup/warehouses')
      .then((d) => {
        setWarehouses(d.warehouses);
        setWarehousesError(null);
      })
      .catch((e: unknown) => {
        setWarehouses([]);
        setWarehousesError(e instanceof Error ? e.message : 'Could not load SQL warehouses');
      })
      .finally(() => {
        warehousesInFlight.current = false;
        setWarehousesLoading(false);
      });
  };

  useEffect(() => {
    seedConfiguredSchema(promptSchema, setPromptSchemas);
    seedConfiguredSchema(evalSchema, setEvalSchemas);
    loadConfiguredCatalogs({ silent: true });
    apiFetch('/setup/catalogs/warm', { method: 'POST' }).catch(() => {});
    void browseSettingsExperiments();
    if (evaluateTabEnabled) loadWarehouses();
    return () => {
      if (catalogSearchTimer.current) clearTimeout(catalogSearchTimer.current);
      if (promptSchemaSearchTimer.current) clearTimeout(promptSchemaSearchTimer.current);
      if (evalSchemaSearchTimer.current) clearTimeout(evalSchemaSearchTimer.current);
    };
  }, []);

  // After warehouses load, attach sql_warehouse_id when we only have a name (common right after config resolve)
  useEffect(() => {
    if (warehouseId || !warehouseName?.trim() || warehouses.length === 0) return;
    const name = warehouseName.trim();
    const wh = warehouses.find((w) => w.name === name || w.name.trim() === name);
    if (wh) setWarehouseId(wh.id);
  }, [warehouses, warehouseName, warehouseId]);

  const handleCatalogChange = (val: string) => {
    setCatalog(val);
    setPromptSchema('');
    setPromptSchemas([]);
    setPromptSchemasError(null);
    if (val) loadConfiguredPromptSchemas(val);
  };

  const handleEvalCatalogChange = (val: string) => {
    setEvalCatalog(val);
    setEvalSchema('');
    setEvalSchemas([]);
    setEvalSchemasError(null);
    if (val) loadConfiguredEvalSchemas(val);
  };

  const handlePromptSchemaQueryChange = useCallback(
    (query: string) => {
      if (!catalog) return;
      if (promptSchemaSearchTimer.current) clearTimeout(promptSchemaSearchTimer.current);
      const trimmed = query.trim();
      if (trimmed.length < 2) {
        loadConfiguredPromptSchemas(catalog);
        return;
      }
      promptSchemaSearchTimer.current = setTimeout(() => {
        searchPromptSchemas(catalog, trimmed);
      }, 300);
    },
    [catalog, loadConfiguredPromptSchemas, searchPromptSchemas],
  );

  const handleEvalSchemaQueryChange = useCallback(
    (query: string) => {
      if (!evalCatalog) return;
      if (evalSchemaSearchTimer.current) clearTimeout(evalSchemaSearchTimer.current);
      const trimmed = query.trim();
      if (trimmed.length < 2) {
        loadConfiguredEvalSchemas(evalCatalog);
        return;
      }
      evalSchemaSearchTimer.current = setTimeout(() => {
        searchEvalSchemas(evalCatalog, trimmed);
      }, 300);
    },
    [evalCatalog, loadConfiguredEvalSchemas, searchEvalSchemas],
  );

  const resolveWarehouseId = () => {
    if (warehouseId) return warehouseId;
    const n = warehouseName?.trim();
    if (!n) return '';
    const wh = warehouses.find((w) => w.name === n || w.name.trim() === n);
    return wh?.id ?? '';
  };

  const handleSave = async () => {
    if (!catalog) { setError('Please select a catalog.'); return; }
    if (!promptSchema?.trim()) { setError('Please enter or select a prompt schema.'); return; }
    if (evaluateTabEnabled) {
      if (!evalCatalog?.trim()) { setError('Please select an eval dataset catalog.'); return; }
      if (!evalSchema?.trim()) { setError('Please enter or select an eval dataset schema.'); return; }
    }
    let wid = '';
    if (evaluateTabEnabled) {
      wid = resolveWarehouseId();
      if (!wid) {
        setError('Please select a SQL warehouse from the list (wait for warehouses to load if needed).');
        return;
      }
    }
    setSaving(true);
    setError(null);
    try {
      invalidateCache('config');
      const updated = await apiFetch<AppConfig>('/config', {
        method: 'POST',
        body: JSON.stringify({
          prompt_catalog: catalog,
          prompt_schema: promptSchema.trim(),
          eval_catalog: evalCatalog.trim(),
          eval_schema: evalSchema.trim(),
          sql_warehouse_id: wid || config.sql_warehouse_id || '',
          sql_warehouse_name: evaluateTabEnabled
            ? (warehouseName?.trim() || warehouses.find((w) => w.id === wid)?.name || '')
            : (config.sql_warehouse_name || ''),
          evaluate_tab_enabled: evaluateTabEnabled,
          mlflow_experiment_name: mlflowExperiment.trim(),
        }),
      });
      invalidateCache('config');
      onSave(updated);
    } catch (e: any) {
      setError(e.message ?? 'Failed to save settings.');
    } finally {
      setSaving(false);
    }
  };

  // Sync warehouseName from loaded list when it resolves (covers first-time setup)
  useEffect(() => {
    const resolved = warehouses.find((w) => w.id === warehouseId)?.name;
    if (resolved) setWarehouseName(resolved);
  }, [warehouses, warehouseId]);


  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/20 z-40" onClick={onClose} />

      {/* Slide-over panel */}
      <div className="fixed inset-y-0 right-0 z-50 w-full max-w-md bg-white shadow-2xl flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
          <div>
            <h2 className="text-base font-semibold text-gray-900">App Settings</h2>
            <p className="text-xs text-gray-500 mt-0.5">Changes are saved to the server and apply to all users.</p>
          </div>
          <button onClick={onClose} className="p-1.5 rounded hover:bg-gray-100 text-gray-400 hover:text-gray-600">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-6 py-5 space-y-6">

          {/* Prompt Registry section */}
          <div className="space-y-4">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Prompt Registry</h3>

            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5">
                <label className="text-sm font-medium text-gray-700">Catalog</label>
                <div className="relative group">
                  <Info className="w-3.5 h-3.5 text-gray-400 cursor-help" />
                  <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 w-64 px-3 py-2 bg-gray-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-10">
                    The Unity Catalog that contains your prompts.
                    <div className="absolute right-full top-1/2 -translate-y-1/2 border-4 border-transparent border-r-gray-900" />
                  </div>
                </div>
              </div>
              <SearchableSelect
                value={catalog}
                onChange={handleCatalogChange}
                options={catalogOptions}
                placeholder="Select a catalog..."
                allowClear={false}
                onOpen={handleCatalogOpen}
                onQueryChange={handleCatalogQueryChange}
                loading={catalogsLoading}
                loadingLabel="Searching catalogs…"
                emptyHint="Type at least 2 characters to search catalogs"
                minSearchChars={2}
              />
              {catalogsError && (
                <p className="text-xs text-red-600">{catalogsError}</p>
              )}
            </div>

            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5">
                <label className="text-sm font-medium text-gray-700">Schema</label>
                <div className="relative group">
                  <Info className="w-3.5 h-3.5 text-gray-400 cursor-help" />
                  <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 w-64 px-3 py-2 bg-gray-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-10">
                    Schema where your prompts are registered.
                    <div className="absolute right-full top-1/2 -translate-y-1/2 border-4 border-transparent border-r-gray-900" />
                  </div>
                </div>
              </div>
              <SearchableSelect
                value={promptSchema}
                onChange={setPromptSchema}
                options={promptSchemaOptions}
                placeholder="Select a schema…"
                allowClear={false}
                onOpen={() => {
                  seedConfiguredSchema(promptSchema, setPromptSchemas);
                  loadConfiguredPromptSchemas(catalog);
                }}
                onQueryChange={handlePromptSchemaQueryChange}
                loading={promptSchemasLoading}
                loadingLabel="Searching schemas…"
                emptyHint={catalog ? 'Type at least 2 characters to search schemas' : 'Select a catalog first'}
                minSearchChars={2}
              />
              {promptSchemasError && (
                <p className="text-xs text-red-600">{promptSchemasError}</p>
              )}
            </div>

            <div className="space-y-1.5">
              <label className="text-sm font-medium text-gray-700">Default experiment (optional)</label>
              <p className="text-xs text-gray-500">
                Most users pick an experiment from the header dropdown. Set a deploy-wide default here if needed.
              </p>
              <SearchableSelect
                value={mlflowExperiment}
                onChange={setMlflowExperiment}
                options={settingsExperimentOptions}
                placeholder="Search experiments…"
                allowClear
                onOpen={onSettingsExperimentOpen}
                onQueryChange={onSettingsExperimentQueryChange}
                loading={settingsExperimentsLoading}
                loadingLabel="Searching experiments…"
                emptyHint="Type part of the name to search (e.g. hinge)"
                minSearchChars={2}
              />
              {settingsExperimentsError && (
                <p className="text-xs text-red-600">{settingsExperimentsError}</p>
              )}
            </div>
          </div>

          <hr className="border-gray-100" />

          {/* Evaluation section */}
          <div className="space-y-4">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Evaluate Tab</h3>

            <label className="flex items-start gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={evaluateTabEnabled}
                onChange={(e) => setEvaluateTabEnabled(e.target.checked)}
                className="mt-0.5 h-4 w-4 rounded border-gray-300 text-databricks-red focus:ring-databricks-red"
              />
              <span className="text-sm text-gray-700 font-medium">Show Evaluate tab</span>
            </label>

            {evaluateTabEnabled && (
              <p className="text-xs text-amber-900 bg-amber-50 border border-amber-100 rounded-lg px-3 py-2">
                <strong className="font-semibold">Experimental.</strong> Evaluate has not been fully tested — use for
                exploration only and verify results before relying on them.
              </p>
            )}

            {evaluateTabEnabled && (
              <>
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide pt-1">Compute</h3>

            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5">
                <label className="text-sm font-medium text-gray-700">SQL Warehouse</label>
                <div className="relative group">
                  <Info className="w-3.5 h-3.5 text-gray-400 cursor-help" />
                  <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 w-64 px-3 py-2 bg-gray-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-10">
                    Used to read evaluation datasets. Warehouses auto-resume if suspended — no need to start them manually.
                    <div className="absolute right-full top-1/2 -translate-y-1/2 border-4 border-transparent border-r-gray-900" />
                  </div>
                </div>
              </div>
              <SearchableSelect
                value={warehouseName}
                onChange={(name) => {
                  const wh = warehouses.find(
                    (w) => w.name === name || w.name.trim() === name.trim(),
                  );
                  if (wh) {
                    setWarehouseId(wh.id);
                    setWarehouseName(wh.name);
                  } else if (name.trim()) {
                    setWarehouseName(name.trim());
                  }
                }}
                options={warehouses.map((w) => ({ value: w.name, label: w.name }))}
                placeholder="Select a warehouse..."
                allowClear={false}
                onOpen={loadWarehouses}
                loading={warehousesLoading}
              />
              {warehousesError && (
                <p className="text-xs text-amber-800 bg-amber-50 border border-amber-100 rounded-lg px-3 py-2">
                  <strong className="font-medium">Could not load warehouses.</strong> {warehousesError} Open this
                  dropdown again to retry. For local dev, run{' '}
                  <code className="text-[11px] bg-amber-100 px-1 rounded">databricks auth login</code> for the same
                  workspace the app uses.
                </p>
              )}
            </div>

            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide pt-1">Evaluation Data</h3>

            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5">
                <label className="text-sm font-medium text-gray-700">Catalog</label>
                <div className="relative group">
                  <Info className="w-3.5 h-3.5 text-gray-400 cursor-help" />
                  <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 w-64 px-3 py-2 bg-gray-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-10">
                    Unity Catalog containing your evaluation datasets.
                    <div className="absolute right-full top-1/2 -translate-y-1/2 border-4 border-transparent border-r-gray-900" />
                  </div>
                </div>
              </div>
              <SearchableSelect
                value={evalCatalog}
                onChange={handleEvalCatalogChange}
                options={catalogOptions}
                placeholder="Select a catalog..."
                allowClear={false}
                onOpen={handleCatalogOpen}
                onQueryChange={handleCatalogQueryChange}
                loading={catalogsLoading}
                loadingLabel="Searching catalogs…"
                emptyHint="Type at least 2 characters to search catalogs"
                minSearchChars={2}
              />
            </div>

            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5">
                <label className="text-sm font-medium text-gray-700">Schema</label>
                <div className="relative group">
                  <Info className="w-3.5 h-3.5 text-gray-400 cursor-help" />
                  <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 w-64 px-3 py-2 bg-gray-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-10">
                    Schema where your evaluation datasets are stored.
                    <div className="absolute right-full top-1/2 -translate-y-1/2 border-4 border-transparent border-r-gray-900" />
                  </div>
                </div>
              </div>
              <SearchableSelect
                value={evalSchema}
                onChange={setEvalSchema}
                options={evalSchemaOptions}
                placeholder="Select a schema…"
                allowClear={false}
                onOpen={() => {
                  seedConfiguredSchema(evalSchema, setEvalSchemas);
                  loadConfiguredEvalSchemas(evalCatalog);
                }}
                onQueryChange={handleEvalSchemaQueryChange}
                loading={evalSchemasLoading}
                loadingLabel="Searching schemas…"
                emptyHint={evalCatalog ? 'Type at least 2 characters to search schemas' : 'Select a catalog first'}
                minSearchChars={2}
              />
              {evalSchemasError && (
                <p className="text-xs text-red-600">{evalSchemasError}</p>
              )}
            </div>
              </>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-200 space-y-2">
          {error && <p className="text-xs text-red-600">{error}</p>}
          <button
            onClick={handleSave}
            disabled={
              saving ||
              !catalog ||
              !promptSchema?.trim() ||
              (evaluateTabEnabled &&
                (!evalCatalog?.trim() ||
                  !evalSchema?.trim() ||
                  !warehouseName?.trim() ||
                  !resolveWarehouseId()))
            }
            className="w-full flex items-center justify-center gap-2 px-4 py-2 text-sm font-medium text-white bg-databricks-red rounded-md hover:bg-databricks-red/90 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
            Save Settings
          </button>
        </div>
      </div>
    </>
  );
}
