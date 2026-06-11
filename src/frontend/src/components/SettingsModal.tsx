import { useState, useEffect, useRef } from 'react';
import { Info, Loader2, Save, X } from 'lucide-react';
import SearchableSelect from './SearchableSelect';
import type { AppConfig } from '../types';
import { apiFetch } from '../hooks/useApi';

interface Props {
  config: AppConfig;
  onSave: (updated: AppConfig) => void;
  onClose: () => void;
}

interface Warehouse {
  id: string;
  name: string;
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

  // Discovery state
  const [catalogs, setCatalogs] = useState<string[]>([]);
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

  /** Only dedupe concurrent in-flight calls — never block retries after failure or when reopening the dropdown */
  const catalogsInFlight = useRef(false);
  const warehousesInFlight = useRef(false);

  const loadCatalogs = () => {
    if (catalogsInFlight.current) return;
    catalogsInFlight.current = true;
    setCatalogsLoading(true);
    setCatalogsError(null);
    apiFetch<{ catalogs: string[] }>('/setup/catalogs')
      .then((d) => {
        setCatalogs(d.catalogs);
        setCatalogsError(null);
      })
      .catch((e: unknown) => {
        setCatalogs([]);
        setCatalogsError(e instanceof Error ? e.message : 'Could not load catalogs');
      })
      .finally(() => {
        catalogsInFlight.current = false;
        setCatalogsLoading(false);
      });
  };

  const loadPromptSchemas = (cat: string) => {
    if (!cat) return;
    setPromptSchemasLoading(true);
    setPromptSchemasError(null);
    apiFetch<{ schemas: string[] }>(`/setup/schemas?${new URLSearchParams({ catalog: cat })}`)
      .then((d) => {
        setPromptSchemas(d.schemas);
        setPromptSchemasError(null);
      })
      .catch((e: unknown) => {
        setPromptSchemas([]);
        setPromptSchemasError(e instanceof Error ? e.message : 'Could not load schemas');
      })
      .finally(() => setPromptSchemasLoading(false));
  };

  const loadEvalSchemas = (cat: string) => {
    if (!cat) return;
    setEvalSchemasLoading(true);
    setEvalSchemasError(null);
    apiFetch<{ schemas: string[] }>(`/setup/schemas?${new URLSearchParams({ catalog: cat })}`)
      .then((d) => {
        setEvalSchemas(d.schemas);
        setEvalSchemasError(null);
      })
      .catch((e: unknown) => {
        setEvalSchemas([]);
        setEvalSchemasError(e instanceof Error ? e.message : 'Could not load schemas');
      })
      .finally(() => setEvalSchemasLoading(false));
  };

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

  // On mount: eager-load discovery lists (always fetch warehouses so name→id can resolve for Save)
  useEffect(() => {
    if (!catalog) loadCatalogs();
    if (catalog && !promptSchema) loadPromptSchemas(catalog);
    if (evalCatalog && !evalSchema) loadEvalSchemas(evalCatalog);
    loadWarehouses();
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
    if (val) loadPromptSchemas(val);
  };

  const handleEvalCatalogChange = (val: string) => {
    setEvalCatalog(val);
    setEvalSchema('');
    setEvalSchemas([]);
    setEvalSchemasError(null);
    if (val) loadEvalSchemas(val);
  };

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
    const wid = resolveWarehouseId();
    if (!wid) {
      setError('Please select a SQL warehouse from the list (wait for warehouses to load if needed).');
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const updated = await apiFetch<AppConfig>('/config', {
        method: 'POST',
        body: JSON.stringify({
          prompt_catalog: catalog,
          prompt_schema: promptSchema.trim(),
          eval_catalog: evalCatalog.trim(),
          eval_schema: evalSchema.trim(),
          sql_warehouse_id: wid,
          sql_warehouse_name: warehouseName?.trim() || warehouses.find((w) => w.id === wid)?.name || '',
          evaluate_tab_enabled: evaluateTabEnabled,
        }),
      });
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

          {/* Compute section */}
          <div className="space-y-4">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Compute</h3>

            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5">
                <label className="text-sm font-medium text-gray-700">SQL Warehouse</label>
                <div className="relative group">
                  <Info className="w-3.5 h-3.5 text-gray-400 cursor-help" />
                  <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 w-64 px-3 py-2 bg-gray-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-10">
                    Used to read your evaluation datasets. Warehouses auto-resume if suspended — no need to start them manually.
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
          </div>

          <hr className="border-gray-100" />

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
                options={catalogs.map((c) => ({ value: c, label: c }))}
                placeholder="Select a catalog..."
                allowClear={false}
                onOpen={loadCatalogs}
                loading={catalogsLoading}
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
                options={promptSchemas.map((s) => ({ value: s, label: s }))}
                placeholder="Select a schema…"
                allowClear={false}
                onOpen={() => loadPromptSchemas(catalog)}
                loading={promptSchemasLoading}
              />
              {promptSchemasError && (
                <p className="text-xs text-red-600">{promptSchemasError}</p>
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
              <span className="text-sm text-gray-700">
                <span className="font-medium">Show Evaluate tab</span>
                <span className="block text-xs text-gray-500 mt-0.5">
                  Off by default. Turn on to expose batch evaluation in the tab bar.
                </span>
              </span>
            </label>

            {evaluateTabEnabled && (
              <p className="text-xs text-amber-900 bg-amber-50 border border-amber-100 rounded-lg px-3 py-2">
                <strong className="font-semibold">Experimental.</strong> Evaluate has not been fully tested — use for
                exploration only and verify results before relying on them.
              </p>
            )}

            {evaluateTabEnabled && (
              <>
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
                options={catalogs.map((c) => ({ value: c, label: c }))}
                placeholder="Select a catalog..."
                allowClear={false}
                onOpen={loadCatalogs}
                loading={catalogsLoading}
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
                options={evalSchemas.map((s) => ({ value: s, label: s }))}
                placeholder="Select a schema…"
                allowClear={false}
                onOpen={() => loadEvalSchemas(evalCatalog)}
                loading={evalSchemasLoading}
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
              !warehouseName?.trim() ||
              !promptSchema?.trim() ||
              (evaluateTabEnabled && (!evalCatalog?.trim() || !evalSchema?.trim())) ||
              !resolveWarehouseId()
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
