import { useEffect } from 'react';
import { RefreshCw, FileText, Tag, Plus } from 'lucide-react';
import { LoadingSpinner } from './LoadingSpinner';
import type { PromptInfo, PromptVersion } from '../types';
import SearchableSelect from './SearchableSelect';

interface Props {
  prompts: PromptInfo[];
  promptsLoading: boolean;
  promptsError: string | null;
  selectedPrompt: string | null;
  onSelectPrompt: (name: string | null) => void;
  versions: PromptVersion[];
  versionsLoading: boolean;
  selectedVersion: string | null;
  onSelectVersion: (version: string | null) => void;
  onRefresh: () => void;
  onCreateNew: () => void;
  onNewVersion?: () => void;
}

export default function PromptSelector({
  prompts,
  promptsLoading,
  promptsError,
  selectedPrompt,
  onSelectPrompt,
  versions,
  versionsLoading,
  selectedVersion,
  onSelectVersion,
  onRefresh,
  onCreateNew,
  onNewVersion,
}: Props) {
  // Auto-select latest version when versions load
  useEffect(() => {
    if (versions.length > 0 && !selectedVersion) {
      onSelectVersion(versions[0].version);
    }
  }, [versions, selectedVersion, onSelectVersion]);

  // Extract short name from fully qualified name for display
  const shortName = (fullName: string) => {
    const parts = fullName.split('.');
    return parts.length > 2 ? parts[parts.length - 1] : fullName;
  };

  return (
    <div className="flex flex-col h-full space-y-4">
      {/* Prompt name selector */}
      <div className="flex-shrink-0">
        <div className="flex items-center justify-between mb-1.5">
          <label className="section-label">Prompt</label>
          <div className="flex items-center gap-1.5">
            <button
              onClick={onCreateNew}
              className="text-gray-400 hover:text-databricks-red transition-colors"
              title="Create new prompt"
            >
              <Plus className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={onRefresh}
              className="inline-flex h-5 w-5 items-center justify-center text-gray-400 hover:text-gray-600 transition-colors"
              title="Refresh prompts"
            >
              {promptsLoading ? (
                <LoadingSpinner className="w-3.5 h-3.5 text-gray-500" />
              ) : (
                <RefreshCw className="w-3.5 h-3.5" />
              )}
            </button>
          </div>
        </div>

        {promptsError && (
          <div className="mb-2 text-xs text-red-600 bg-red-50 rounded-lg px-3 py-2">
            {promptsError}
          </div>
        )}

        <SearchableSelect
          value={selectedPrompt || ''}
          onChange={(val) => {
            onSelectPrompt(val || null);
            onSelectVersion(null);
          }}
          loading={promptsLoading}
          placeholder={
            promptsLoading
              ? 'Loading prompts from registry…'
              : prompts.length === 0
                ? 'No prompts found'
                : 'Select a prompt…'
          }
          options={prompts.map((p) => ({ value: p.name, label: shortName(p.name) }))}
        />

        {selectedPrompt && (
          <div className="mt-1.5">
            <div className="text-[11px] font-mono text-gray-400 mb-0.5">
              {selectedPrompt}
            </div>
            <div className="text-xs text-gray-500">
              {prompts.find((p) => p.name === selectedPrompt)?.description || 'No description'}
            </div>
          </div>
        )}
      </div>

      {/* Version selector */}
      {selectedPrompt && (
        <div className="flex flex-col flex-1 min-h-0">
          <div className="flex items-center justify-between mb-1.5 flex-shrink-0">
            <label className="section-label !mb-0">Version</label>
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
          {versionsLoading ? (
            <div className="text-xs text-gray-500 py-2">Loading versions...</div>
          ) : versions.length === 0 ? (
            <div className="text-xs text-gray-500 py-2">No versions found</div>
          ) : (
            <div className="space-y-1.5 flex-1 overflow-y-auto">
              {versions.map((v) => {
                const isSelected = selectedVersion === v.version;
                return (
                  <button
                    key={v.version}
                    onClick={() => onSelectVersion(v.version)}
                    className={`w-full text-left rounded-lg px-3 py-2 text-sm transition-colors border ${
                      isSelected
                        ? 'border-databricks-red bg-red-50 text-gray-900'
                        : 'border-gray-200 bg-white text-gray-700 hover:border-gray-300 hover:bg-gray-50'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <FileText className="w-3.5 h-3.5 text-gray-400 flex-shrink-0" />
                        <span className="font-medium">v{v.version}</span>
                      </div>
                    </div>
                    {v.description && (
                      <div className="mt-0.5 ml-5 text-[11px] text-gray-500 truncate">
                        {v.description}
                      </div>
                    )}
                    {v.aliases && v.aliases.length > 0 && (
                      <div className="flex gap-1 mt-1 ml-5">
                        {v.aliases.map((alias) => (
                          <span
                            key={alias}
                            className="inline-flex items-center gap-0.5 rounded-full bg-blue-50 px-1.5 py-0.5 text-[10px] font-medium text-blue-700"
                          >
                            <Tag className="w-2.5 h-2.5" />
                            {alias}
                          </span>
                        ))}
                      </div>
                    )}
                    {v.template_preview && (
                      <div className="mt-1 ml-5 text-[11px] text-gray-400 truncate">
                        {v.template_preview.replace(/<\/?(?:system|user)>\n?/g, '').replace(/\\n|\n/g, ' ').trim()}
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
