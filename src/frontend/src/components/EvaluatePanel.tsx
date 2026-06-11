import { useState, useEffect } from 'react';
import { FlaskConical, Loader2, ChevronDown, ChevronUp, Plus, Pencil, Trash2, Eye, RotateCcw, Wand2, X } from 'lucide-react';
import SearchableSelect from './SearchableSelect';
import TemplatePreview from './TemplatePreview';
import type { PromptInfo, PromptVersion, PromptTemplate } from '../types';
import { useEvalTables, useEvalColumns, useTablePreview, useRunEval, useJudges, useDeleteJudge, useJudgeDetail, useEvalHistory } from '../hooks/useEvalApi';
import { parseTemplateVariables } from '../utils/templateUtils';
import JudgeForm from './eval/JudgeForm';
import DatasetTable from './eval/DatasetTable';
import EvalRunHistory from './eval/EvalRunHistory';
import ConfirmDialog from './ConfirmDialog';

const BUILTIN_JUDGES = [
  { value: 'quality', label: 'Quality scorer', description: 'Scores overall response quality using an LLM judge on a 1–5 scale, evaluating helpfulness, accuracy, and completeness.' },
  { value: 'safety', label: 'Safety', description: 'Detects harmful, offensive, or toxic content in responses' },
  { value: 'relevance_to_query', label: 'Relevance to Query', description: 'Checks if the response addresses the input question' },
  { value: 'fluency', label: 'Fluency', description: 'Evaluates clarity and readability of the response' },
  { value: 'completeness', label: 'Completeness', description: 'Assesses whether the response fully answers the question' },
  { value: 'summarization', label: 'Summarization', description: 'Rates the quality of a summarization response' },
  { value: 'correctness', label: 'Correctness', description: 'Checks response against a ground-truth expected answer — requires selecting an expected response column' },
];

interface Props {
  evalCatalog: string;
  evalSchema: string;
  prompts: PromptInfo[];
  versions: PromptVersion[];
  selectedPrompt: string | null;
  selectedVersion: string | null;
  onSelectPrompt: (name: string | null) => void;
  onSelectVersion: (version: string | null) => void;
  models: { name: string; state: string }[];
  selectedModel: string | null;
  onSelectModel: (name: string | null) => void;
  template: PromptTemplate | null;
  experimentName: string;
  onNewVersion?: () => void;
  onExperimentUrl?: (url: string) => void;
  onOpenSettings?: () => void;
  hasWarehouse?: boolean;
}


function JudgePreview({ name }: { name: string }) {
  const [expanded, setExpanded] = useState(false);
  const { detail, loading } = useJudgeDetail(expanded ? name : null);

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden mt-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-2 bg-gray-50 hover:bg-gray-100 transition-colors text-left"
      >
        <Eye className="w-3.5 h-3.5 text-gray-400 flex-shrink-0" />
        <span className="text-xs font-semibold text-gray-600 flex-1">Judge Preview</span>
        {loading && <Loader2 className="w-3 h-3 animate-spin text-gray-400" />}
        {!loading && (expanded
          ? <ChevronUp className="w-3.5 h-3.5 text-gray-400" />
          : <ChevronDown className="w-3.5 h-3.5 text-gray-400" />
        )}
      </button>
      {expanded && (
        <div className="px-3 py-2.5 bg-white">
          {loading ? (
            <p className="text-xs text-gray-400">Loading judge details...</p>
          ) : !detail ? (
            <p className="text-xs text-gray-400 italic">Could not load judge details.</p>
          ) : detail.type === 'guidelines' && detail.guidelines ? (
            <ol className="space-y-2">
              {detail.guidelines.map((g, i) => (
                <li key={i} className="flex gap-2 items-start text-xs text-gray-700">
                  <span className="flex-shrink-0 w-4 h-4 rounded-full bg-purple-100 text-purple-700 text-[10px] flex items-center justify-center font-bold mt-0.5">
                    {i + 1}
                  </span>
                  <span className="leading-relaxed">{g}</span>
                </li>
              ))}
            </ol>
          ) : (
            <p className="text-xs text-gray-700 whitespace-pre-wrap leading-relaxed">{detail.instructions}</p>
          )}
        </div>
      )}
    </div>
  );
}

export default function EvaluatePanel({
  evalCatalog, evalSchema,
  prompts, versions, selectedPrompt, selectedVersion,
  onSelectPrompt, onSelectVersion, models, selectedModel, onSelectModel,
  template, experimentName, onNewVersion, onExperimentUrl, onOpenSettings, hasWarehouse,
}: Props) {
  const [selectedTable, setSelectedTable] = useState<string | null>(null);
  const [columnMapping, setColumnMapping] = useState<Record<string, string>>({});
  const [maxRows, setMaxRows] = useState(5);
  const [scorerName, setScorerName] = useState<string | null>('quality');
  const [expectationsColumn, setExpectationsColumn] = useState<string | null>(null);
  const [judgeModel, setJudgeModel] = useState<string>('');
  const [judgeTemperature, setJudgeTemperature] = useState(0);
  const [useCustomJudgeModel, setUseCustomJudgeModel] = useState(false);
  const [showCreateJudge, setShowCreateJudge] = useState(false);
  const [editingJudge, setEditingJudge] = useState<string | null>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const { deleteJudge, loading: deleteLoading } = useDeleteJudge();

  // Reset table selection when eval catalog/schema changes
  useEffect(() => {
    setSelectedTable(null);
    setColumnMapping({});
  }, [evalCatalog, evalSchema]);

  const { judges, loading: judgesLoading, refresh: refreshJudges } = useJudges(experimentName);
  const { tables, loading: tablesLoading, error: tablesError } = useEvalTables(evalCatalog, evalSchema);
  const { columns } = useEvalColumns(evalCatalog, evalSchema, selectedTable);
  const { columns: previewCols, rows: previewRows, totalRows, loading: previewLoading } = useTablePreview(evalCatalog, evalSchema, selectedTable);
  const { result, loading, error, runEval, abort, reset } = useRunEval();
  const { runs: historyRuns, loading: historyLoading, refresh: refreshHistory } = useEvalHistory(selectedPrompt, experimentName);

  // Bubble experiment URL up to App when eval completes
  useEffect(() => {
    if (result?.experiment_url) onExperimentUrl?.(result.experiment_url);
  }, [result?.experiment_url]);

  // Refresh history after each eval run completes
  useEffect(() => {
    if (result?.run_id) refreshHistory();
  }, [result?.run_id]);

  // Auto-select the latest version when versions load and none is selected.
  // Versions are cleared synchronously on prompt change (via resetVersions in App.tsx),
  // so this only fires once the fresh versions for the current prompt have loaded.
  useEffect(() => {
    if (selectedPrompt && versions.length > 0 && !selectedVersion) {
      onSelectVersion(versions[0].version);
    }
  }, [selectedPrompt, versions, selectedVersion, onSelectVersion]);

  // Use variables from the full template (reliable) or fall back to version preview parsing
  const variables: string[] = template?.variables ?? (() => {
    const selectedVersionData = versions.find((v) => v.version === selectedVersion);
    if (!selectedVersionData?.template_preview) return [];
    return parseTemplateVariables(selectedVersionData.template_preview);
  })();

  // Clamp maxRows to actual dataset size once totalRows is known
  useEffect(() => {
    if (totalRows !== null && maxRows > totalRows) setMaxRows(totalRows);
  }, [totalRows]);

  // Auto-populate mapping when columns load: if variable name exactly matches a column, pre-select it
  useEffect(() => {
    if (columns.length === 0 || variables.length === 0) return;
    setColumnMapping((prev) => {
      const next = { ...prev };
      let changed = false;
      for (const v of variables) {
        if (!next[v] && columns.includes(v)) {
          next[v] = v;
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [columns, variables]);

  const handleReset = () => {
    reset();
    setSelectedTable(null);
    setColumnMapping({});
    setMaxRows(5);
    onSelectModel(null);
    setScorerName('quality');
    setExpectationsColumn(null);
    setJudgeModel('');
    setJudgeTemperature(0);
    setUseCustomJudgeModel(false);
  };

  const canRun = !!(selectedPrompt && selectedVersion && selectedModel && selectedTable &&
    variables.every((v) => columnMapping[v]) &&
    (scorerName !== 'correctness' || !!expectationsColumn));

  const handleRun = async () => {
    if (!canRun || !selectedPrompt || !selectedVersion || !selectedModel || !selectedTable) return;
    reset();
    await runEval({
      prompt_name: selectedPrompt,
      prompt_version: selectedVersion,
      model_name: selectedModel,
      dataset_catalog: evalCatalog,
      dataset_schema: evalSchema,
      dataset_table: selectedTable,
      column_mapping: columnMapping,
      max_rows: maxRows,
      experiment_name: experimentName || undefined,
      scorer_name: (!scorerName || scorerName === 'quality') ? undefined : scorerName,
      judge_model: judgeModel || undefined,
      judge_temperature: judgeTemperature,
      expectations_column: expectationsColumn || undefined,
    });
  };

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <div
        role="status"
        className="shrink-0 px-4 py-2.5 bg-amber-50 border-b border-amber-100 text-xs text-amber-950"
      >
        <strong className="font-semibold">Experimental — not fully tested.</strong> Batch evaluation is enabled for
        exploration; verify scores and behavior before relying on results in production workflows.
      </div>
      <div className="flex-1 flex overflow-hidden min-h-0">
      {/* Left config panel */}
      <div className="w-1/3 min-w-[272px] flex-shrink-0 border-r border-gray-200 bg-white flex flex-col overflow-hidden">
      {!hasWarehouse && evalCatalog && (
        <div className="flex-shrink-0 px-4 py-2.5 border-b border-amber-100 bg-amber-50">
          <p className="text-xs text-amber-700">
            SQL warehouse not configured.{' '}
            {onOpenSettings && (
              <button onClick={onOpenSettings} className="underline font-medium">
                Open Settings →
              </button>
            )}
          </p>
        </div>
      )}
      <div className="flex-1 overflow-y-auto p-4 space-y-5">

        {/* ── 1. Prompt ───────────────────────────────── */}

        {/* Prompt picker */}
        <div>
          <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">Prompt</label>
          <SearchableSelect
            value={selectedPrompt || ''}
            onChange={(val) => { onSelectPrompt(val || null); onSelectVersion(null); setColumnMapping({}); reset(); }}
            placeholder="Select a prompt..."
            options={prompts.map((p) => ({ value: p.name, label: p.name.split('.').pop() ?? p.name }))}
          />
        </div>

        {/* Eval history — prompt level, all versions */}
        {selectedPrompt && (
          <EvalRunHistory
            promptName={selectedPrompt}
            promptVersion={selectedVersion}
            runs={historyRuns}
            loading={historyLoading}
          />
        )}

        {/* Version picker */}
        {selectedPrompt && (
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wide">Version</label>
              {onNewVersion && (
                <button
                  onClick={onNewVersion}
                  className="flex items-center gap-1 text-[11px] font-medium text-databricks-red hover:text-red-700 transition-colors"
                  title="Create a new version based on the current template"
                >
                  <Plus className="w-3 h-3" />
                  New version
                </button>
              )}
            </div>
            <SearchableSelect
              value={selectedVersion || ''}
              onChange={(val) => { onSelectVersion(val || null); setColumnMapping({}); reset(); }}
              placeholder="Select a version..."
              options={versions.map((v) => ({
                value: v.version,
                label: `v${v.version}${v.aliases.length > 0 ? ` · ${v.aliases.join(', ')}` : ''}${v.description ? ` — ${v.description}` : ''}`,
              }))}
            />
          </div>
        )}

        {/* Prompt template preview */}
        {template && selectedVersion && (
          <TemplatePreview template={template} />
        )}

        {/* ── 2. Dataset + Column Mapping ─────────────── */}
        <hr className="border-gray-100" />

        {/* Eval dataset — catalog, schema, and table in one group */}
        <div>
          <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">Evaluation Data</label>
          <div className="mb-2">
            {evalCatalog ? (
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-gray-500 truncate">
                  {evalCatalog}<span className="text-gray-300">.</span>{evalSchema}
                </span>
                {onOpenSettings && (
                  <button onClick={onOpenSettings} className="text-xs text-databricks-red hover:underline flex-shrink-0 ml-2">
                    Edit →
                  </button>
                )}
              </div>
            ) : (
              onOpenSettings && (
                <button onClick={onOpenSettings} className="text-xs text-databricks-red hover:underline">
                  Open Settings →
                </button>
              )
            )}
          </div>
          {tablesLoading ? (
            <p className="text-xs text-gray-400">Loading tables...</p>
          ) : tablesError ? (
            <p className="text-xs text-red-500" title={tablesError}>Failed to load tables — check catalog/schema and warehouse config.</p>
          ) : (
            <SearchableSelect
              value={selectedTable || ''}
              onChange={(val) => { setSelectedTable(val || null); setColumnMapping({}); reset(); }}
              placeholder="Select a table..."
              options={tables.map((t) => ({ value: t, label: t }))}
            />
          )}
          <div className="mt-3">
            <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
              Max Rows <span className="font-normal text-gray-400">({maxRows}{totalRows !== null ? ` of ${totalRows}` : ''})</span>
            </label>
            <input
              type="range" min={1} max={totalRows ?? 20} value={maxRows}
              onChange={(e) => setMaxRows(Number(e.target.value))}
              className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-databricks-red"
            />
          </div>
        </div>

        {/* Column mapping — inline under Dataset once a table is selected and prompt has variables */}
        {selectedTable && variables.length > 0 && (
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Map Variables → Columns
              </label>
              <div className="flex items-center gap-1.5">
                <button
                  onClick={() => {
                    setColumnMapping((prev) => {
                      const next = { ...prev };
                      for (const v of variables) {
                        if (columns.includes(v)) next[v] = v;
                      }
                      return next;
                    });
                  }}
                  className="flex items-center gap-0.5 text-[11px] text-databricks-red hover:text-red-700 font-medium"
                  title="Auto-map variables to columns with matching names"
                >
                  <Wand2 className="w-3 h-3" /> Auto
                </button>
                <button
                  onClick={() => setColumnMapping({})}
                  className="text-[11px] text-gray-400 hover:text-gray-600"
                  title="Clear all column mappings"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            </div>
            <div className="space-y-2">
              {[...variables].sort((a, b) => {
                const posA = columns.indexOf(a);
                const posB = columns.indexOf(b);
                return (posA === -1 ? 999 : posA) - (posB === -1 ? 999 : posB);
              }).map((v) => (
                <div key={v} className="space-y-1">
                  <span className="text-xs font-mono text-purple-700 bg-purple-50 px-1.5 py-0.5 rounded inline-block max-w-full truncate">{`{{${v}}}`}</span>
                  <SearchableSelect
                    value={columnMapping[v] || ''}
                    onChange={(val) => setColumnMapping((prev) => ({ ...prev, [v]: val }))}
                    placeholder="Select a column..."
                    options={columns.map((c) => ({ value: c, label: c }))}
                  />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── 3. Model Endpoint ──────────────────────── */}
        <hr className="border-gray-100" />

        <div>
          <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">Model Endpoint</label>
          <SearchableSelect
            value={selectedModel || ''}
            onChange={(val) => onSelectModel(val)}
            placeholder="Select a model..."
            options={models.filter((m) => m.state === 'READY').map((m) => ({ value: m.name, label: m.name }))}
          />
        </div>

        {/* ── 4. Judge / Scorer ──────────────────────── */}
        <hr className="border-gray-100" />

        {/* Judge (registered scorer) picker */}
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wide">Judge</label>
            <div className="flex items-center gap-1.5">
              {scorerName && !BUILTIN_JUDGES.some((b) => b.value === scorerName) && (
                <button
                  onClick={() => {
                    setEditingJudge(scorerName);
                    setShowCreateJudge(true);
                  }}
                  className="text-[11px] text-blue-600 hover:text-blue-800 font-medium"
                  title="Edit judge"
                >
                  <Pencil className="w-3 h-3" />
                </button>
              )}
              {scorerName && !BUILTIN_JUDGES.some((b) => b.value === scorerName) && (
                <button
                  onClick={() => setShowDeleteConfirm(true)}
                  disabled={deleteLoading}
                  className="text-[11px] text-red-500 hover:text-red-700 font-medium"
                  title="Delete judge"
                >
                  <Trash2 className="w-3 h-3" />
                </button>
              )}
              <button
                onClick={() => {
                  setEditingJudge(null);
                  setShowCreateJudge(true);
                }}
                className="flex items-center gap-0.5 text-[11px] text-databricks-red hover:text-red-700 font-medium"
              >
                <Plus className="w-3 h-3" /> New
              </button>
            </div>
          </div>
          <SearchableSelect
            value={scorerName || ''}
            onChange={(val) => {
              setScorerName(val || null);
              if (val !== 'correctness') setExpectationsColumn(null);
            }}
            disabled={judgesLoading}
            placeholder="Select a judge..."
            groups={[
              { label: 'Built-in Presets', options: BUILTIN_JUDGES },
              ...(judges.length > 0
                ? [{ label: 'Registered Judges', options: judges.map((j) => ({ value: j.name, label: j.name })) }]
                : []),
            ]}
          />

          {/* Description for selected built-in judge */}
          {BUILTIN_JUDGES.some((b) => b.value === scorerName) && (
            <p className="mt-1.5 text-[11px] text-gray-400 leading-relaxed">
              {BUILTIN_JUDGES.find((b) => b.value === scorerName)?.description}
            </p>
          )}

          {/* Preview for registered (custom) judges */}
          {scorerName && !BUILTIN_JUDGES.some((b) => b.value === scorerName) && (
            <JudgePreview key={scorerName} name={scorerName} />
          )}

          {/* Expectations column — only shown when Correctness scorer is selected */}
          {scorerName === 'correctness' && (
            <div className="mt-3 pl-3 border-l-2 border-gray-200">
              <label className="block text-xs font-medium text-gray-500 mb-1.5">
                Expected response column
              </label>
              <SearchableSelect
                value={expectationsColumn || ''}
                onChange={(val) => setExpectationsColumn(val || null)}
                placeholder="Select a column..."
                options={columns.map((c) => ({ value: c, label: c }))}
              />
              <p className="mt-1 text-[11px] text-gray-400">
                The column containing the ground-truth expected response for each row.
              </p>
            </div>
          )}

          {/* Judge model + temperature — only relevant for the quality scorer */}
          {scorerName === 'quality' && (
            <div className="mt-3 space-y-3 pl-3 border-l-2 border-gray-200">
              <div>
                <label className="flex items-center gap-2 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={useCustomJudgeModel}
                    onChange={(e) => {
                      setUseCustomJudgeModel(e.target.checked);
                      if (!e.target.checked) setJudgeModel('');
                    }}
                    className="rounded border-gray-300 accent-databricks-red"
                  />
                  <span className="text-xs text-gray-500">Use a different model for judging</span>
                </label>
                {useCustomJudgeModel && (
                  <div className="mt-2">
                    <SearchableSelect
                      value={judgeModel || ''}
                      onChange={(val) => setJudgeModel(val)}
                      placeholder="Select a model..."
                      options={models.filter((m) => m.state === 'READY').map((m) => ({ value: m.name, label: m.name }))}
                    />
                  </div>
                )}
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-500 mb-1.5">
                  Judge temperature <span className="text-gray-400">({judgeTemperature.toFixed(1)})</span>
                </label>
                <input
                  type="range" min={0} max={1} step={0.1} value={judgeTemperature}
                  onChange={(e) => setJudgeTemperature(Number(e.target.value))}
                  className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-databricks-red"
                />
                <p className="mt-1 text-[11px] text-gray-400">
                  Lower = more consistent scores. Keep at 0 unless you want varied judgments.
                </p>
              </div>
            </div>
          )}
        </div>

      </div>{/* end scrollable config */}

      {/* ── Sticky Run Footer ──────────────────────────── */}
      <div className="flex-shrink-0 p-4 border-t border-gray-200 bg-white space-y-2">
        <div className="flex gap-2">
          {loading ? (
            <button
              onClick={abort}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 border border-databricks-red text-databricks-red text-sm font-medium rounded-lg hover:bg-red-50 transition-colors"
            >
              <X className="w-4 h-4" /> Stop
            </button>
          ) : (
            <button
              onClick={handleRun}
              disabled={!canRun}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-databricks-red text-white text-sm font-medium rounded-lg disabled:opacity-40 disabled:cursor-not-allowed hover:bg-red-700 transition-colors"
            >
              <FlaskConical className="w-4 h-4" /> Run Evaluation
            </button>
          )}
          <button
            onClick={handleReset}
            className="px-3 py-2.5 border border-gray-200 rounded-lg text-gray-500 hover:text-gray-700 hover:border-gray-300 transition-colors"
            title="Reset to defaults"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>

        {!canRun && !loading && (
          <p className="text-xs text-gray-400 text-center">
            {!selectedPrompt ? 'Select a prompt' :
             !selectedVersion ? 'Select a version' :
             !selectedModel ? 'Select a model' :
             !selectedTable ? 'Select a dataset' :
             !variables.every((v) => columnMapping[v]) ? 'Map all variables to columns' :
             scorerName === 'correctness' && !expectationsColumn ? 'Select an expected response column' :
             ''}
          </p>
        )}
      </div>
      </div>{/* end left panel */}

      {/* Right panel — dataset preview + results */}
      <div className="flex-1 overflow-hidden flex flex-col">
        {error && (
          <div className="bg-red-50 border border-red-200 border-b-0 px-4 py-3 text-sm text-red-700 flex-shrink-0">
            {error}
          </div>
        )}
        {selectedTable ? (
          <DatasetTable
            previewColumns={previewCols}
            previewRows={previewRows}
            previewLoading={previewLoading}
            variables={variables}
            columnMapping={columnMapping}
            result={result}
            loading={loading}
            maxRows={maxRows}
            scorerName={scorerName}
            template={template}
          />
        ) : (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <FlaskConical className="w-10 h-10 text-gray-300 mx-auto mb-3" />
              <p className="text-sm text-gray-500">Configure the evaluation and click Run</p>
              <p className="text-xs text-gray-400 mt-1">Each row in the dataset will be run through the prompt and scored with <code className="bg-gray-100 px-1 rounded">mlflow.evaluate()</code></p>
            </div>
          </div>
        )}
      </div>
      </div>

      {/* Delete judge confirmation */}
      {showDeleteConfirm && scorerName && (
        <ConfirmDialog
          title={`Delete "${scorerName}"?`}
          message="This judge will be permanently removed and can't be recovered."
          confirmLabel="Delete Judge"
          loading={deleteLoading}
          onConfirm={async () => {
            try {
              await deleteJudge({ name: scorerName, experiment_name: experimentName || undefined });
              setScorerName(null);
              refreshJudges();
            } catch { /* error in hook */ }
            setShowDeleteConfirm(false);
          }}
          onCancel={() => setShowDeleteConfirm(false)}
        />
      )}

      {/* Judge create/edit slide-over — rendered outside both panels so it overlays the full tab */}
      {showCreateJudge && (
        <JudgeForm
          experimentName={experimentName}
          editingJudge={editingJudge}
          onSaved={(name) => {
            setScorerName(name);
            refreshJudges();
            if (!editingJudge) {
              // New judge: close the form
              setShowCreateJudge(false);
              setEditingJudge(null);
            }
            // Edit: form stays open showing the success banner
          }}
          onCancel={() => {
            setShowCreateJudge(false);
            setEditingJudge(null);
          }}
        />
      )}
    </div>
  );
}
