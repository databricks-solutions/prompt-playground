import { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import TabBar from './components/TabBar';
import type { Tab } from './components/TabBar';
import ConfirmDialog from './components/ConfirmDialog';
import PromptSelector from './components/PromptSelector';
import VariableInputs from './components/VariableInputs';
import ModelSelector from './components/ModelSelector';
import RunControls from './components/RunControls';
import PromptPreview from './components/PromptPreview';
import PromptDiffView from './components/PromptDiffView';
import ResponsePanel from './components/ResponsePanel';
import EvaluatePanel from './components/EvaluatePanel';
import HowToTab from './components/HowToTab';
import PromptForm from './components/PromptForm';
import SettingsModal from './components/SettingsModal';
import SearchableSelect from './components/SearchableSelect';
import {
  useConfig,
  usePrompts,
  usePromptVersions,
  usePromptTemplate,
  useModels,
  useExperiments,
  useExperimentPrompts,
  useRunPrompt,
  useSaveVersion,
  useCreatePrompt,
} from './hooks/useApi';
import { usePromptEditor } from './hooks/usePromptEditor';

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('prompts');
  const [pendingTab, setPendingTab] = useState<Tab | null>(null);
  const [pendingPromptChange, setPendingPromptChange] = useState<{ name: string | null } | null>(null);
  const [pendingVersionChange, setPendingVersionChange] = useState<{ version: string | null } | null>(null);
  const [showCreatePrompt, setShowCreatePrompt] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showDiff, setShowDiff] = useState(false);

  // Load catalog/schema config from backend (set via app.yaml env vars)
  const { config, loading: configLoading, refresh: refreshConfig, isConfigured } = useConfig();

  // Catalog / schema state — initialized from config once loaded
  const [catalog, setCatalog] = useState('');
  const [schema, setSchema] = useState('');

  // Experiment name state
  const [experimentName, setExperimentName] = useState('');
  const [filterByExperiment, setFilterByExperiment] = useState(true);

  // Sync catalog/schema/experiment from config once it loads
  useEffect(() => {
    if (config && !catalog) {
      setCatalog(config.prompt_catalog);
      setSchema(config.prompt_schema);
    }
    if (config && !experimentName) {
      setExperimentName(config.mlflow_experiment_name);
    }
  }, [config]);

  // Auto-open settings on first load if app is unconfigured
  useEffect(() => {
    if (!configLoading && !isConfigured) {
      setShowSettings(true);
    }
  }, [configLoading, isConfigured]);

  // Use '' as fallback (not 'main') so usePrompts won't fire until config is loaded
  const activeCatalog = catalog || config?.prompt_catalog || '';
  const activeSchema = schema || config?.prompt_schema || '';

  // Prompt state
  const [selectedPrompt, setSelectedPrompt] = useState<string | null>(null);
  const [selectedVersion, setSelectedVersion] = useState<string | null>(null);
  const [variableValues, setVariableValues] = useState<Record<string, string>>({});

  // Model state
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [maxTokens, setMaxTokens] = useState(4096);
  const [temperature, setTemperature] = useState(1.0);

  // API hooks
  const {
    prompts,
    loading: promptsLoading,
    error: promptsError,
    refresh: refreshPrompts,
  } = usePrompts(activeCatalog, activeSchema);
  const { versions, loading: versionsLoading, refresh: refreshVersions, reset: resetVersions } = usePromptVersions(selectedPrompt);
  const { template, loading: templateLoading } = usePromptTemplate(
    selectedPrompt,
    selectedVersion
  );
  const {
    models,
    loading: modelsLoading,
    error: modelsError,
    refresh: refreshModels,
    ensureLoaded: ensureModelsLoaded,
  } = useModels();
  const { experiments, loading: experimentsLoading } = useExperiments();
  const { promptNames: experimentPromptNames, loading: experimentPromptsLoading, refresh: refreshExperimentPrompts } = useExperimentPrompts(experimentName, activeCatalog, activeSchema);
  const filteredPrompts = (filterByExperiment && experimentPromptNames)
    ? prompts.filter((p) => experimentPromptNames.includes(p.name) || p.name === selectedPrompt)
    : prompts;
  // Don't show the selected prompt in the browser if it's been filtered out by the experiment toggle
  const promptsTabSelectedPrompt = selectedPrompt;
  const { result, loading: runLoading, error: runError, run, reset } = useRunPrompt();
  const [experimentUrl, setExperimentUrl] = useState<string | undefined>(undefined);

  // Set experiment URL from the selected experiment (on load and on change)
  useEffect(() => {
    if (!experimentName) return;
    const exp = experiments.find(e => e.name === experimentName);
    if (exp?.url) setExperimentUrl(exp.url);
  }, [experimentName, experiments]);

  // Update experiment URL when playground run completes (fallback)
  useEffect(() => {
    if (result?.experiment_url) setExperimentUrl(result.experiment_url);
  }, [result?.experiment_url]);
  // Auto-select latest version when versions load and none is selected
  useEffect(() => {
    if (versions.length > 0 && !selectedVersion) {
      setSelectedVersion(versions[0].version);
    }
  }, [versions, selectedVersion]);

  const { save: saveVersion, loading: saveLoading, error: saveError } = useSaveVersion();
  const { create: createPrompt, loading: createLoading, error: createError } = useCreatePrompt();

  const editor = usePromptEditor({
    template,
    selectedPrompt,
    activeCatalog,
    activeSchema,
    createPrompt,
    saveVersion,
    refreshPrompts,
    refreshVersions,
    setSelectedPrompt,
    setSelectedVersion,
    setVariableValues,
  });

  const handleExperimentChange = useCallback((name: string) => {
    setExperimentName(name);
    setFilterByExperiment(true);
  }, []);

  // Handle prompt selection - clear variables, versions, and exit edit mode
  const handleSelectPrompt = useCallback((name: string | null) => {
    if (editor.isDirty) { setPendingPromptChange({ name }); return; }
    resetVersions();
    setSelectedPrompt(name);
    setSelectedVersion(null);
    setVariableValues({});
    setShowDiff(false);
    editor.exitEdit();
    if (name) ensureModelsLoaded();
  }, [editor.isDirty, editor.exitEdit, resetVersions, ensureModelsLoaded]);

  const handleTabChange = useCallback((tab: Tab) => {
    if (editor.isDirty) {
      setPendingTab(tab);
    } else {
      if (editor.isEditing) editor.exitEdit();
      setActiveTab(tab);
    }
  }, [editor.isDirty, editor.isEditing, editor.exitEdit]);

  // Handle version selection - keep variable values if same variables exist
  const handleSelectVersion = useCallback((version: string | null) => {
    if (editor.isDirty) { setPendingVersionChange({ version }); return; }
    setSelectedVersion(version);
    editor.exitEdit();
  }, [editor.isDirty, editor.exitEdit]);

  // Handle variable value change
  const handleVariableChange = useCallback((key: string, value: string) => {
    setVariableValues((prev) => ({ ...prev, [key]: value }));
  }, []);

  // Run prompt
  const handleRun = useCallback(
    async (settings: { max_tokens: number; temperature: number }) => {
      if (!selectedModel) return;

      const promptName = selectedPrompt || `${activeCatalog}.${activeSchema}.${editor.newPromptName || 'draft'}`;
      const promptVersion = selectedVersion || 'draft';

      if (!editor.isEditing && (!selectedPrompt || !selectedVersion)) return;

      try {
        await run({
          prompt_name: promptName,
          prompt_version: promptVersion,
          variables: variableValues,
          model_name: selectedModel,
          max_tokens: settings.max_tokens,
          temperature: settings.temperature,
          experiment_name: experimentName || undefined,
          draft_template: editor.isEditing ? editor.draftTemplate : undefined,
        });
      } catch {
        // Error is captured in the hook
      }
    },
    [selectedPrompt, selectedVersion, selectedModel, variableValues, run,
     editor.isEditing, editor.draftTemplate, activeCatalog, activeSchema, editor.newPromptName, experimentName]
  );

  const handleReset = useCallback(() => {
    reset();
    setVariableValues({});
  }, [reset]);

  const canRun = !!(
    selectedModel &&
    !templateLoading &&
    (
      (!editor.isEditing && selectedPrompt && selectedVersion) ||
      (editor.isEditing && editor.draftTemplate.trim().length > 0)
    )
  );

  const unfilledVars = editor.activeVariables.filter(v => !variableValues[v]?.trim());

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      <Header
        experimentName={experimentName}
        experiments={experiments}
        experimentsLoading={experimentsLoading}
        onExperimentChange={handleExperimentChange}
        onOpenSettings={() => setShowSettings(true)}
        filterByExperiment={filterByExperiment}
        onFilterChange={setFilterByExperiment}
        experimentPromptsLoading={experimentPromptsLoading}
        filteredCount={filteredPrompts.length}
        totalCount={prompts.length}
      />

      <TabBar activeTab={activeTab} onTabChange={handleTabChange} experimentUrl={experimentUrl} />


      <main className="flex-1 overflow-hidden">
        {activeTab === 'howto' && <HowToTab />}

        {/* Evaluate tab */}
        <div className={activeTab !== 'evaluate' ? 'hidden' : 'h-full'}>
          <EvaluatePanel
            evalCatalog={config?.eval_catalog ?? activeCatalog}
            evalSchema={config?.eval_schema ?? 'eval_data'}
            prompts={filteredPrompts}
            versions={versions}
            selectedPrompt={selectedPrompt}
            selectedVersion={selectedVersion}
            onSelectPrompt={handleSelectPrompt}
            onSelectVersion={handleSelectVersion}
            models={models}
            selectedModel={selectedModel}
            onSelectModel={setSelectedModel}
            template={template}
            experimentName={experimentName}
            onNewVersion={selectedVersion ? () => { editor.toggleEdit(); setActiveTab('prompts'); } : undefined}
            onExperimentUrl={setExperimentUrl}
            onOpenSettings={() => setShowSettings(true)}
            hasWarehouse={!!config?.sql_warehouse_id}
          />
        </div>

        {/* Prompts tab — browse, manage, edit */}
        <div className={`h-full flex ${activeTab !== 'prompts' ? 'hidden' : ''}`}>
          {/* Left Panel - Prompt browser */}
          <div className="w-1/3 min-w-[280px] flex-shrink-0 border-r border-gray-200 bg-white flex flex-col overflow-hidden">
            {/* Prompt Registry badge — click to open Settings */}
            <div className="flex-shrink-0 px-4 py-2.5 border-b border-gray-100">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Prompt Registry</span>
                {activeCatalog && (
                  <button
                    onClick={() => setShowSettings(true)}
                    className="text-xs text-databricks-red hover:underline flex-shrink-0"
                  >
                    Edit →
                  </button>
                )}
              </div>
              {activeCatalog ? (
                <span className="text-xs font-mono text-gray-400 truncate block mt-0.5">
                  {activeCatalog}<span className="text-gray-300">.</span>{activeSchema}
                </span>
              ) : (
                <button
                  onClick={() => setShowSettings(true)}
                  className="text-xs text-databricks-red hover:underline mt-0.5 block"
                >
                  Configure prompt registry →
                </button>
              )}
            </div>

            {/* Warehouse warning — shown when catalog is set but warehouse is missing */}
            {activeCatalog && !config?.sql_warehouse_id && (
              <div className="flex-shrink-0 px-4 py-2.5 border-b border-amber-100 bg-amber-50">
                <p className="text-xs text-amber-700">
                  SQL warehouse not configured.{' '}
                  <button onClick={() => setShowSettings(true)} className="underline font-medium">
                    Open Settings →
                  </button>
                </p>
              </div>
            )}

            {/* Flexible middle — PromptSelector grows to fill */}
            <div className="flex-1 overflow-hidden p-4">
              <PromptSelector
                prompts={filteredPrompts}
                promptsLoading={promptsLoading}
                promptsError={promptsError}
                selectedPrompt={promptsTabSelectedPrompt}
                onSelectPrompt={handleSelectPrompt}
                versions={versions}
                versionsLoading={versionsLoading}
                selectedVersion={selectedVersion}
                onSelectVersion={handleSelectVersion}
                onRefresh={() => { refreshPrompts(); refreshExperimentPrompts(); }}
                onCreateNew={() => setShowCreatePrompt(true)}
                onNewVersion={selectedVersion ? editor.toggleEdit : undefined}
              />
            </div>

            {/* Sticky footer — Test in Playground */}
            <div className="flex-shrink-0 p-4 border-t border-gray-200 bg-white">
              <button
                onClick={() => handleTabChange('playground')}
                disabled={!selectedPrompt || !selectedVersion}
                className="w-full py-2 px-4 rounded-md text-sm font-medium bg-databricks-red text-white hover:bg-red-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                Test in Playground →
              </button>
            </div>
          </div>

          {/* Right Panel - PromptPreview or PromptDiffView */}
          <div className="flex-1 flex flex-col overflow-hidden">
            <div className="flex-1 overflow-hidden">
              {showDiff && selectedPrompt ? (
                <PromptDiffView
                  promptName={selectedPrompt}
                  versions={versions}
                  currentVersion={selectedVersion}
                  onClose={() => setShowDiff(false)}
                />
              ) : (
                <PromptPreview
                  template={editor.activeTemplate}
                  variables={editor.activeVariables}
                  values={variableValues}
                  isEditing={editor.isEditing}
                  isLatestVersion={!versions.length || !selectedVersion || selectedVersion === versions[0]?.version}
                  onToggleEdit={editor.toggleEdit}
                  draftTemplate={editor.draftTemplate}
                  onDraftChange={editor.setDraftTemplate}
                  onDraftVariablesChange={editor.setDraftVariables}
                  onSave={editor.save}
                  saveLoading={saveLoading || createLoading}
                  saveError={saveError || createError}
                  isDirty={editor.isDirty}
                  onCompare={() => setShowDiff(true)}
                  canCompare={versions.length > 1}
                  nextVersion={versions.length > 0 ? String(Number(versions[0].version) + 1) : '1'}
                />
              )}
            </div>
          </div>
        </div>

        {/* Playground tab — fill variables, run, iterate */}
        <div className={`h-full flex ${activeTab !== 'playground' ? 'hidden' : ''}`}>
          {/* Left Panel - Run controls */}
          <div className="w-1/3 min-w-[280px] flex-shrink-0 border-r border-gray-200 bg-white flex flex-col overflow-hidden">
            <div className="flex-1 overflow-y-auto p-4 space-y-6">
              {/* Loaded prompt indicator */}
              <div>
                <label className="section-label">Prompt</label>
                {selectedPrompt ? (
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-700">
                        {selectedPrompt.split('.').pop()}
                      </span>
                      <button
                        onClick={() => handleTabChange('prompts')}
                        className="text-xs text-databricks-red hover:underline"
                      >
                        Change →
                      </button>
                    </div>
                    <SearchableSelect
                      value={selectedVersion || ''}
                      onChange={(val) => handleSelectVersion(val || null)}
                      disabled={versionsLoading}
                      loading={versionsLoading}
                      allowClear={false}
                      placeholder="Select version..."
                      options={versions.map((v) => ({
                        value: v.version,
                        label: `v${v.version}${v.aliases?.length ? ` · ${v.aliases.join(', ')}` : ''}${v.description ? ` — ${v.description}` : ''}`,
                      }))}
                    />
                  </div>
                ) : (
                  <button
                    onClick={() => setActiveTab('prompts')}
                    className="text-sm text-databricks-red hover:underline"
                  >
                    Select a prompt →
                  </button>
                )}
              </div>

              {editor.activeVariables.length > 0 && (
                <VariableInputs
                  variables={editor.activeVariables}
                  values={variableValues}
                  onChange={handleVariableChange}
                />
              )}

              <ModelSelector
                models={models}
                loading={modelsLoading}
                error={modelsError}
                selectedModel={selectedModel}
                onSelectModel={setSelectedModel}
                onRefresh={refreshModels}
                maxTokens={maxTokens}
                temperature={temperature}
                onMaxTokensChange={setMaxTokens}
                onTemperatureChange={setTemperature}
              />
            </div>

            {/* Sticky Run Footer */}
            <div className="flex-shrink-0 p-4 border-t border-gray-200 bg-white">
              <RunControls
                canRun={canRun}
                loading={runLoading}
                onRun={handleRun}
                onReset={handleReset}
                unfilledVars={unfilledVars}
                maxTokens={maxTokens}
                temperature={temperature}
              />
            </div>
          </div>

          {/* Right Panel - Preview & Response */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {/* Preview (top) */}
            <div className="h-3/5 min-h-[200px] border-b border-gray-200 overflow-hidden">
              <PromptPreview
                template={editor.activeTemplate}
                variables={editor.activeVariables}
                values={variableValues}
                isEditing={editor.isEditing}
                isLatestVersion={!versions.length || !selectedVersion || selectedVersion === versions[0]?.version}
                onToggleEdit={editor.toggleEdit}
                draftTemplate={editor.draftTemplate}
                onDraftChange={editor.setDraftTemplate}
                onDraftVariablesChange={editor.setDraftVariables}
                onSave={async (desc) => { await editor.save(desc); reset(); }}
                saveLoading={saveLoading || createLoading}
                saveError={saveError || createError}
                isDirty={editor.isDirty}
              />
            </div>

            {/* Response (bottom) */}
            <div className="flex-1 overflow-hidden">
              <ResponsePanel result={result} loading={runLoading} error={runError} />
            </div>
          </div>
        </div>
      </main>

      {pendingTab && (
        <ConfirmDialog
          title="Discard unsaved changes?"
          message="You have unsaved edits to this version. Leave without saving?"
          confirmLabel="Discard"
          variant="warning"
          onConfirm={() => {
            editor.exitEdit();
            setActiveTab(pendingTab);
            setPendingTab(null);
          }}
          onCancel={() => setPendingTab(null)}
        />
      )}

      {pendingVersionChange !== null && (
        <ConfirmDialog
          title="Discard unsaved changes?"
          message="You have unsaved edits to this version. Switching versions will discard them."
          confirmLabel="Discard"
          variant="warning"
          onConfirm={() => {
            const { version } = pendingVersionChange;
            setPendingVersionChange(null);
            editor.exitEdit();
            setSelectedVersion(version);
          }}
          onCancel={() => setPendingVersionChange(null)}
        />
      )}

      {pendingPromptChange !== null && (
        <ConfirmDialog
          title="Discard unsaved changes?"
          message="You have unsaved edits to this version. Switching prompts will discard them."
          confirmLabel="Discard"
          variant="warning"
          onConfirm={() => {
            const { name } = pendingPromptChange;
            setPendingPromptChange(null);
            editor.exitEdit();
            resetVersions();
            setSelectedPrompt(name);
            setSelectedVersion(null);
            setVariableValues({});
          }}
          onCancel={() => setPendingPromptChange(null)}
        />
      )}

      {showCreatePrompt && (
        <PromptForm
          catalog={activeCatalog}
          schema={activeSchema}
          experimentName={experimentName}
          onSaved={async (name, version) => {
            setShowCreatePrompt(false);
            await refreshPrompts();
            await refreshExperimentPrompts();
            setSelectedPrompt(name);
            setSelectedVersion(version);
            setVariableValues({});
            editor.exitEdit();
          }}
          onCancel={() => setShowCreatePrompt(false)}
        />
      )}

      {showSettings && config && (
        <SettingsModal
          config={config}
          onSave={(updated) => {
            refreshConfig();
            setCatalog(updated.prompt_catalog);
            setSchema(updated.prompt_schema);
            setExperimentName(updated.mlflow_experiment_name);
            setSelectedPrompt(null);
            setSelectedVersion(null);
            setVariableValues({});
            setShowSettings(false);
          }}
          onClose={() => setShowSettings(false)}
        />
      )}
    </div>
  );
}
