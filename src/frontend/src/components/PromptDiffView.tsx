import { Loader2 } from 'lucide-react';
import { useState, useMemo, useEffect } from 'react';
import type { PromptVersion } from '../types';
import { usePromptTemplate } from '../hooks/usePromptApi';
import { computeLineDiff, getTemplateText } from '../utils/diffUtils';
import { parseSystemUser } from '../utils/templateUtils';
import DiffView, { DiffLines } from './DiffView';

type ViewMode = 'chat' | 'raw';

interface Props {
  promptName: string;
  versions: PromptVersion[];
  currentVersion: string | null;
  onClose: () => void;
}

export default function PromptDiffView({ promptName, versions, currentVersion, onClose }: Props) {
  // Default: A = previous version (base/older), B = current version (newer)
  // versions[] is sorted descending (newest first), so versions[0] is latest
  const currentIdx = versions.findIndex((v) => v.version === currentVersion);
  const defaultB = currentVersion ?? versions[0]?.version ?? '';
  // Pick the version just before current as the base; fall back to the oldest available
  const defaultA =
    currentIdx >= 0 && currentIdx < versions.length - 1
      ? versions[currentIdx + 1].version
      : versions.find((v) => v.version !== defaultB)?.version ?? defaultB;

  const [versionA, setVersionA] = useState(defaultA);
  const [versionB, setVersionB] = useState(defaultB);
  const [viewMode, setViewMode] = useState<ViewMode>('raw');

  const { template: tplA, loading: loadingA } = usePromptTemplate(promptName, versionA || null);
  const { template: tplB, loading: loadingB } = usePromptTemplate(promptName, versionB || null);
  const loading = loadingA || loadingB;

  const rawA = tplA ? getTemplateText(tplA) : '';
  const rawB = tplB ? getTemplateText(tplB) : '';
  const hasSystemA = tplA ? !!parseSystemUser(rawA).system : false;
  const hasSystemB = tplB ? !!parseSystemUser(rawB).system : false;
  const canChat = hasSystemA || hasSystemB;

  // Auto-set view mode when templates load
  useEffect(() => {
    if (!loading && (tplA || tplB)) {
      setViewMode(canChat ? 'chat' : 'raw');
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tplA?.version, tplB?.version]);

  const diffContent = useMemo(() => {
    if (!tplA || !tplB) return null;
    if (viewMode === 'raw') {
      return { type: 'raw' as const, lines: computeLineDiff(rawA, rawB) };
    }
    const { system: sysA, user: userA } = parseSystemUser(rawA);
    const { system: sysB, user: userB } = parseSystemUser(rawB);
    return {
      type: 'chat' as const,
      systemLines: computeLineDiff(sysA ?? '', sysB ?? ''),
      userLines: computeLineDiff(userA, userB),
    };
  }, [tplA, tplB, viewMode, rawA, rawB]);

  const onlyOneVersion = versions.length <= 1;

  return (
    <div className="card h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-gray-100 flex-wrap">
        {/* Left: view mode toggle */}
        <div className="flex rounded-md border border-gray-200 overflow-hidden text-xs">
          <button
            onClick={onClose}
            className="px-3 py-1.5 font-medium transition-colors text-gray-500 hover:text-gray-700"
          >
            Preview
          </button>
          <button className="px-3 py-1.5 font-medium border-l border-gray-200 bg-gray-100 text-gray-800 cursor-default">
            Compare
          </button>
        </div>

        {/* Version selectors */}
        <div className="flex items-center gap-1.5 ml-auto">
          <select
            value={versionA}
            onChange={(e) => setVersionA(e.target.value)}
            className="border border-gray-200 rounded-md px-2 py-1 text-gray-700 bg-white focus:outline-none focus:ring-1 focus:ring-databricks-red text-xs font-mono"
          >
            {versions.map((v) => (
              <option key={v.version} value={v.version}>
                v{v.version}{v.aliases.length ? ` (${v.aliases.join(', ')})` : ''}
              </option>
            ))}
          </select>
          <span className="text-gray-400 text-xs">→</span>
          <select
            value={versionB}
            onChange={(e) => setVersionB(e.target.value)}
            disabled={onlyOneVersion}
            className="border border-gray-200 rounded-md px-2 py-1 text-gray-700 bg-white focus:outline-none focus:ring-1 focus:ring-databricks-red text-xs font-mono disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {versions.map((v) => (
              <option key={v.version} value={v.version}>
                v{v.version}{v.aliases.length ? ` (${v.aliases.join(', ')})` : ''}
              </option>
            ))}
          </select>
        </div>

        {/* Chat/Raw toggle */}
        <div className="flex rounded-md border border-gray-200 overflow-hidden text-xs">
          <button
            onClick={() => setViewMode('chat')}
            disabled={!canChat}
            className={`px-3 py-1.5 font-medium transition-colors ${
              viewMode === 'chat'
                ? 'bg-gray-100 text-gray-800'
                : 'text-gray-500 hover:text-gray-700'
            } disabled:opacity-40 disabled:cursor-not-allowed`}
          >
            Chat
          </button>
          <button
            onClick={() => setViewMode('raw')}
            className={`px-3 py-1.5 font-medium transition-colors border-l border-gray-200 ${
              viewMode === 'raw'
                ? 'bg-gray-100 text-gray-800'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            Raw
          </button>
        </div>

      </div>

      {/* Body */}
      <div className="flex-1 p-4 overflow-auto">
        {onlyOneVersion ? (
          <p className="text-sm text-gray-400 italic">Only one version exists — nothing to compare.</p>
        ) : loading ? (
          <div className="flex items-center gap-2 text-xs text-gray-400 py-4">
            <Loader2 className="w-3.5 h-3.5 animate-spin" /> Loading templates...
          </div>
        ) : !diffContent ? (
          <p className="text-xs text-gray-400 italic">Select two versions to compare.</p>
        ) : diffContent.type === 'raw' ? (
          <DiffView lines={diffContent.lines} />
        ) : (
          <div className="space-y-3">
            {/* System section */}
            <div className="rounded-lg border border-indigo-100 overflow-hidden font-mono text-[11px] leading-relaxed">
              <div className="px-3 py-1.5 bg-indigo-50 border-b border-indigo-100">
                <span className="text-[10px] font-semibold text-indigo-500 uppercase tracking-widest">System</span>
              </div>
              <DiffLines lines={diffContent.systemLines} />
            </div>
            {/* User section */}
            <div className="rounded-lg border border-gray-200 overflow-hidden font-mono text-[11px] leading-relaxed">
              <div className="px-3 py-1.5 bg-gray-50 border-b border-gray-100">
                <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-widest">User</span>
              </div>
              <DiffLines lines={diffContent.userLines} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
