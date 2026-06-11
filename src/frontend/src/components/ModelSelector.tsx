import { RefreshCw, Cpu, Settings2, ChevronDown } from 'lucide-react';
import { LoadingSpinner } from './LoadingSpinner';
import { useState } from 'react';
import type { ModelEndpoint } from '../types';
import SearchableSelect from './SearchableSelect';

interface Props {
  models: ModelEndpoint[];
  loading: boolean;
  error: string | null;
  selectedModel: string | null;
  onSelectModel: (name: string | null) => void;
  onRefresh: () => void;
  maxTokens: number;
  temperature: number;
  onMaxTokensChange: (v: number) => void;
  onTemperatureChange: (v: number) => void;
}

export default function ModelSelector({
  models,
  loading,
  error,
  selectedModel,
  onSelectModel,
  onRefresh,
  maxTokens,
  temperature,
  onMaxTokensChange,
  onTemperatureChange,
}: Props) {
  const [showSettings, setShowSettings] = useState(false);

  // Filter to show only READY endpoints, but keep all if none are ready
  const readyModels = models.filter((m) => m.state === 'READY');
  const displayModels = readyModels.length > 0 ? readyModels : models;

  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <label className="section-label">Model Endpoint</label>
        <button
          onClick={onRefresh}
          className="inline-flex h-5 w-5 items-center justify-center text-gray-400 hover:text-gray-600 transition-colors"
          title="Refresh models"
        >
          {loading ? (
            <LoadingSpinner className="w-3.5 h-3.5 text-gray-500" />
          ) : (
            <RefreshCw className="w-3.5 h-3.5" />
          )}
        </button>
      </div>

      {error && (
        <div className="mb-2 text-xs text-red-600 bg-red-50 rounded-lg px-3 py-2">{error}</div>
      )}

      <SearchableSelect
        value={selectedModel || ''}
        onChange={(val) => onSelectModel(val || null)}
        disabled={loading}
        placeholder={loading ? 'Loading endpoints...' : 'Select a model...'}
        options={displayModels.map((m) => ({ value: m.name, label: m.name }))}
        leftIcon={<Cpu className="w-4 h-4 text-gray-400" />}
      />

      {selectedModel && (
        <div className="mt-1.5 text-xs text-gray-500">
          Task:{' '}
          {displayModels.find((m) => m.name === selectedModel)?.task || 'unknown'}
        </div>
      )}

      {/* Model settings — shown below the selector */}
      <div className="mt-2">
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-700 transition-colors"
        >
          <Settings2 className="w-3.5 h-3.5" />
          <span>Model settings</span>
          <ChevronDown className={`w-3 h-3 transition-transform duration-150 ${showSettings ? 'rotate-180' : ''}`} />
        </button>

        {showSettings && (
          <div className="mt-2 space-y-3 p-3 bg-gray-50 rounded-lg">
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs font-medium text-gray-600">Max Tokens</label>
                <span className="text-xs text-gray-500">{maxTokens}</span>
              </div>
              <input
                type="range"
                min={256}
                max={16384}
                step={256}
                value={maxTokens}
                onChange={(e) => onMaxTokensChange(Number(e.target.value))}
                className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-databricks-red"
              />
            </div>
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs font-medium text-gray-600">Temperature</label>
                <span className="text-xs text-gray-500">{temperature.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.1}
                value={temperature}
                onChange={(e) => onTemperatureChange(Number(e.target.value))}
                className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-databricks-red"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
