import { useState, useMemo } from 'react';
import { X, Loader2, ExternalLink, ChevronDown, ChevronUp, ChevronRight } from 'lucide-react';
import type { EvalHistoryRun, EvalTraceRow, PromptTemplate } from '../../types';
import { usePromptTemplate } from '../../hooks/usePromptApi';
import { useRunTraces } from '../../hooks/useEvalApi';

// ===== UTILITIES =====

type DiffLine = { type: 'added' | 'removed' | 'unchanged'; text: string };

function computeLineDiff(oldText: string, newText: string): DiffLine[] {
  const a = oldText.split('\n');
  const b = newText.split('\n');
  const m = a.length, n = b.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++)
      dp[i][j] = a[i - 1] === b[j - 1] ? dp[i - 1][j - 1] + 1 : Math.max(dp[i - 1][j], dp[i][j - 1]);
  const result: DiffLine[] = [];
  let i = m, j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && a[i - 1] === b[j - 1]) {
      result.unshift({ type: 'unchanged', text: a[i - 1] });
      i--; j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      result.unshift({ type: 'added', text: b[j - 1] });
      j--;
    } else {
      result.unshift({ type: 'removed', text: a[i - 1] });
      i--;
    }
  }
  return result;
}

function getTemplateText(tpl: PromptTemplate): string {
  return (tpl.raw_template ?? tpl.template).replace(/\\n/g, '\n');
}

function relativeTime(ms: number): string {
  const diff = Date.now() - ms;
  const mins = Math.floor(diff / 60000);
  if (mins < 2) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function versionAvg(runs: EvalHistoryRun[], version: string): number | null {
  const scores = runs
    .filter((r) => r.prompt_version === version)
    .map((r) => r.avg_score)
    .filter((s): s is number => s !== null);
  return scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : null;
}

function highlight(text: string): string {
  const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  return esc(text).replace(
    /\{\{\s*(\w+)\s*\}\}/g,
    (_, key) => `<span class="inline-block bg-purple-100 text-purple-700 rounded px-0.5 font-mono">{{${key}}}</span>`
  );
}

// ===== SCORE BADGE =====

function ScoreBadge({ score }: { score: number | string | null }) {
  if (score === null || score === undefined) return null;
  const num = typeof score === 'string' ? parseFloat(score) : score;
  const color = !isNaN(num)
    ? num >= 4 ? 'bg-green-100 text-green-700'
      : num >= 3 ? 'bg-amber-100 text-amber-700'
      : 'bg-red-100 text-red-700'
    : 'bg-gray-100 text-gray-600';
  return (
    <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded ${color}`}>
      {typeof num === 'number' && !isNaN(num) ? num.toFixed(1) : score}
    </span>
  );
}

// ===== SCORE DELTA =====

function ScoreDelta({ current, previous, prevVersion }: {
  current: number | null;
  previous: number | null;
  prevVersion: string | null;
}) {
  if (current === null || previous === null || !prevVersion) return null;
  const delta = current - previous;
  if (Math.abs(delta) < 0.05) {
    return <span className="text-[11px] text-gray-400">unchanged from v{prevVersion}</span>;
  }
  const up = delta > 0;
  return (
    <span className={`text-[11px] font-medium ${up ? 'text-green-600' : 'text-red-500'}`}>
      {up ? '↑' : '↓'} {Math.abs(delta).toFixed(2)} from v{prevVersion}
    </span>
  );
}

// ===== DIFF VIEW =====

function DiffView({ lines }: { lines: DiffLine[] }) {
  return (
    <div className="rounded-lg border border-gray-200 overflow-hidden font-mono text-[11px] leading-relaxed">
      {lines.map((line, i) => {
        const bg =
          line.type === 'added' ? 'bg-green-50' :
          line.type === 'removed' ? 'bg-red-50' : 'bg-white';
        const prefix =
          line.type === 'added' ? '+' :
          line.type === 'removed' ? '-' : ' ';
        const prefixColor =
          line.type === 'added' ? 'text-green-600 select-none' :
          line.type === 'removed' ? 'text-red-500 select-none' :
          'text-transparent select-none';
        const textColor =
          line.type === 'added' ? 'text-green-800' :
          line.type === 'removed' ? 'text-red-700' :
          'text-gray-400';
        return (
          <div key={i} className={`flex px-3 py-0.5 ${bg}`}>
            <span className={`w-4 flex-shrink-0 ${prefixColor}`}>{prefix}</span>
            <span
              className={`flex-1 whitespace-pre-wrap break-words ${textColor}`}
              dangerouslySetInnerHTML={{ __html: highlight(line.text) }}
            />
          </div>
        );
      })}
    </div>
  );
}

// ===== CHANGES SECTION =====
// Fetches both templates and shows a line diff (or full prompt for first version).
// Auto-expands when there are changes; collapses when unchanged.

interface ChangesSectionProps {
  selectedVersion: string;
  prevVersion: string | null;
  promptName: string;
}

function ChangesSection({ selectedVersion, prevVersion, promptName }: ChangesSectionProps) {
  const { template: currentTpl, loading: currentLoading } = usePromptTemplate(promptName, selectedVersion);
  const { template: prevTpl, loading: prevLoading } = usePromptTemplate(promptName, prevVersion);
  const [openOverride, setOpenOverride] = useState<boolean | null>(null);

  const diff = useMemo(() => {
    if (!currentTpl || !prevTpl) return null;
    return computeLineDiff(getTemplateText(prevTpl), getTemplateText(currentTpl));
  }, [currentTpl, prevTpl]);

  const hasChanges = diff ? diff.some((l) => l.type !== 'unchanged') : null;
  const changeCount = diff ? diff.filter((l) => l.type !== 'unchanged').length : 0;
  const loading = currentLoading || prevLoading;

  // Auto-open when there are changes; user can override
  const isOpen = openOverride !== null ? openOverride : (hasChanges ?? !!prevVersion === false);

  const headerLabel = !prevVersion
    ? 'Prompt (first version)'
    : loading || hasChanges === null
      ? `Changes from v${prevVersion}`
      : hasChanges
        ? `Changes from v${prevVersion}  ·  ${changeCount} line${changeCount !== 1 ? 's' : ''} changed`
        : `Prompt (unchanged from v${prevVersion})`;

  return (
    <div>
      <button
        onClick={() => setOpenOverride(!isOpen)}
        className="flex items-center gap-1.5 text-[11px] font-semibold text-gray-400 uppercase tracking-wide mb-2 hover:text-gray-600 transition-colors w-full text-left"
      >
        {isOpen
          ? <ChevronDown className="w-3 h-3 flex-shrink-0" />
          : <ChevronRight className="w-3 h-3 flex-shrink-0" />
        }
        <span className="flex-1">{headerLabel}</span>
        {loading && <Loader2 className="w-3 h-3 animate-spin" />}
      </button>

      {isOpen && (
        loading ? (
          <div className="flex items-center gap-2 text-xs text-gray-400 py-4">
            <Loader2 className="w-3.5 h-3.5 animate-spin" /> Loading templates...
          </div>
        ) : !prevVersion && currentTpl ? (
          // First version — show full prompt
          <div className="rounded-lg border border-gray-200 overflow-hidden">
            <div
              className="px-3 py-2.5 font-mono text-[11px] text-gray-600 whitespace-pre-wrap leading-relaxed"
              dangerouslySetInnerHTML={{ __html: highlight(getTemplateText(currentTpl)) }}
            />
          </div>
        ) : diff && hasChanges ? (
          <DiffView lines={diff} />
        ) : diff && !hasChanges ? (
          <p className="text-xs text-gray-400 italic py-1">Prompt template unchanged from v{prevVersion}.</p>
        ) : (
          <p className="text-xs text-gray-400 italic py-1">Could not load templates.</p>
        )
      )}
    </div>
  );
}

// ===== RUN SUMMARY =====
// Aggregated judge output — score distribution or per-guideline pass rates.
// Shown when a run card is expanded, instead of per-row trace data.

function RunSummary({ rows, loading, error }: { rows: EvalTraceRow[]; loading: boolean; error: string | null }) {
  if (loading) return (
    <div className="flex items-center gap-2 py-2 text-xs text-gray-400">
      <Loader2 className="w-3 h-3 animate-spin" /> Loading...
    </div>
  );
  if (error) return <p className="py-2 text-xs text-red-500">Failed to load trace data.</p>;
  if (rows.length === 0) return <p className="py-2 text-xs text-gray-400 italic">No trace data found.</p>;

  const isGuidelines = rows.some((r) => Array.isArray(r.details) && (r.details?.length ?? 0) > 0);

  if (isGuidelines) {
    const guidelineMap = new Map<string, { passed: number; total: number }>();
    rows.forEach((row) => {
      row.details?.forEach((d) => {
        const name = d.name.includes('/') ? d.name.split('/').pop()! : d.name;
        if (!guidelineMap.has(name)) guidelineMap.set(name, { passed: 0, total: 0 });
        const stat = guidelineMap.get(name)!;
        stat.total++;
        const passed =
          d.value !== null && (
            typeof d.value === 'boolean' ? d.value :
            typeof d.value === 'number' ? d.value >= 1 :
            ['true', 'yes', 'pass', '1'].includes(String(d.value).toLowerCase())
          );
        if (passed) stat.passed++;
      });
    });
    return (
      <div className="space-y-1.5 py-1">
        {[...guidelineMap.entries()].map(([name, { passed, total }]) => {
          const rate = passed / total;
          const barColor = rate >= 0.8 ? 'bg-green-400' : rate >= 0.5 ? 'bg-amber-400' : 'bg-red-400';
          return (
            <div key={name} className="flex items-center gap-2">
              <span className="text-[11px] text-gray-600 flex-1 truncate" title={name}>{name}</span>
              <span className="text-[10px] text-gray-400 flex-shrink-0">{passed}/{total}</span>
              <div className="w-16 h-1.5 bg-gray-100 rounded-full overflow-hidden flex-shrink-0">
                <div className={`h-full rounded-full ${barColor}`} style={{ width: `${rate * 100}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  // Quality/numeric judge — score distribution
  const scores = rows
    .map((r) => {
      if (r.score === null || r.score === undefined) return null;
      const n = typeof r.score === 'string' ? parseFloat(r.score) : r.score;
      return isNaN(n) ? null : n;
    })
    .filter((s): s is number => s !== null);

  if (scores.length === 0) return <p className="py-2 text-xs text-gray-400 italic">No scores recorded.</p>;

  const distribution = new Map<number, number>();
  scores.forEach((s) => {
    const k = Math.round(s);
    distribution.set(k, (distribution.get(k) ?? 0) + 1);
  });
  const maxCount = Math.max(...distribution.values());
  const sortedKeys = [...distribution.keys()].sort((a, b) => b - a);

  return (
    <div className="space-y-1 py-1">
      {sortedKeys.map((score) => {
        const count = distribution.get(score)!;
        const barColor = score >= 4 ? 'bg-green-400' : score >= 3 ? 'bg-amber-400' : 'bg-red-400';
        return (
          <div key={score} className="flex items-center gap-2">
            <span className="text-[11px] text-gray-500 w-4 flex-shrink-0 text-right">{score}</span>
            <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
              <div className={`h-full rounded-full ${barColor}`} style={{ width: `${(count / maxCount) * 100}%` }} />
            </div>
            <span className="text-[10px] text-gray-400 w-8 flex-shrink-0 text-right">{count}</span>
          </div>
        );
      })}
    </div>
  );
}

// ===== RUN DETAIL =====

function RunDetail({ run }: { run: EvalHistoryRun }) {
  const [expanded, setExpanded] = useState(false);
  const { rows, loading, error } = useRunTraces(expanded ? run.run_id : null);

  return (
    <div className="border border-gray-100 rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 px-3 py-2.5 bg-gray-50 hover:bg-gray-100 transition-colors text-left"
      >
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-[11px] text-gray-500">{relativeTime(run.created_at)}</span>
            <ScoreBadge score={run.avg_score} />
            {run.total_rows !== null && (
              <span className="text-[10px] text-gray-400">{run.total_rows} rows</span>
            )}
          </div>
          <div className="text-xs text-gray-600 truncate mt-0.5">
            {run.model || '—'}
            {run.dataset && <span className="text-gray-400"> · {run.dataset.split('.').pop()}</span>}
            {run.scorer && <span className="text-gray-400"> · {run.scorer}</span>}
          </div>
        </div>
        <a
          href={run.run_url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-gray-400 hover:text-databricks-red flex-shrink-0"
          title="View in Databricks"
          onClick={(e) => e.stopPropagation()}
        >
          <ExternalLink className="w-3.5 h-3.5" />
        </a>
        {loading
          ? <Loader2 className="w-3.5 h-3.5 animate-spin text-gray-400 flex-shrink-0" />
          : expanded
            ? <ChevronUp className="w-3.5 h-3.5 text-gray-400 flex-shrink-0" />
            : <ChevronDown className="w-3.5 h-3.5 text-gray-400 flex-shrink-0" />
        }
      </button>
      {expanded && (
        <div className="px-3 border-t border-gray-100 bg-white">
          <RunSummary rows={rows} loading={loading} error={error} />
        </div>
      )}
    </div>
  );
}

// ===== MAIN MODAL =====

interface Props {
  promptName: string;
  currentVersion: string | null;
  runs: EvalHistoryRun[];
  onClose: () => void;
}

export default function EvalHistoryModal({ promptName, currentVersion, runs, onClose }: Props) {
  const availableVersions = [
    ...new Set(runs.map((r) => r.prompt_version).filter(Boolean)),
  ].sort((a, b) => {
    const na = Number(a), nb = Number(b);
    if (!isNaN(na) && !isNaN(nb)) return nb - na;
    return b.localeCompare(a);
  });

  const defaultVersion =
    currentVersion && availableVersions.includes(currentVersion)
      ? currentVersion
      : (availableVersions[0] ?? null);

  const [selectedVersion, setSelectedVersion] = useState<string | null>(defaultVersion);

  const selectedIdx = selectedVersion ? availableVersions.indexOf(selectedVersion) : -1;
  const prevVersion =
    selectedIdx >= 0 && selectedIdx < availableVersions.length - 1
      ? availableVersions[selectedIdx + 1]
      : null;

  const versionRuns = runs.filter((r) => r.prompt_version === selectedVersion);
  const shortName = promptName.split('.').pop() ?? promptName;
  const selectedAvg = selectedVersion ? versionAvg(runs, selectedVersion) : null;
  const prevAvg = prevVersion ? versionAvg(runs, prevVersion) : null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-6" onClick={onClose}>
      <div className="absolute inset-0 bg-black/40" />
      <div
        className="relative bg-white rounded-xl shadow-2xl w-full max-w-4xl flex flex-col overflow-hidden"
        style={{ height: 'min(85vh, 750px)' }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center gap-3 px-5 py-3.5 border-b border-gray-200 flex-shrink-0">
          <h2 className="text-sm font-semibold text-gray-800 flex-1">
            Eval History — <span className="font-mono text-gray-500">{shortName}</span>
          </h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 transition-colors" title="Close">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="flex flex-1 overflow-hidden min-h-0">
          {/* Left: version list */}
          <div className="w-36 flex-shrink-0 border-r border-gray-200 overflow-y-auto bg-gray-50">
            {availableVersions.map((v) => {
              const vRuns = runs.filter((r) => r.prompt_version === v);
              const avg = versionAvg(runs, v);
              const isCurrent = v === currentVersion;
              const isSelected = v === selectedVersion;
              return (
                <button
                  key={v}
                  onClick={() => setSelectedVersion(v)}
                  className={`w-full text-left px-3 py-2.5 border-b border-gray-200 transition-colors ${
                    isSelected
                      ? 'bg-white border-l-2 border-l-databricks-red'
                      : 'hover:bg-white border-l-2 border-l-transparent'
                  }`}
                >
                  <div className="flex items-center gap-1.5 mb-0.5">
                    <span className={`text-xs font-semibold ${isSelected ? 'text-databricks-red' : 'text-gray-700'}`}>
                      v{v}
                    </span>
                    {isCurrent && (
                      <span className="text-[9px] bg-blue-100 text-blue-600 px-1 py-0.5 rounded font-medium leading-none">
                        current
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-1.5">
                    {avg !== null && <ScoreBadge score={avg.toFixed(1)} />}
                    <span className="text-[10px] text-gray-400">
                      {vRuns.length} run{vRuns.length !== 1 ? 's' : ''}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>

          {/* Right: version detail */}
          <div className="flex-1 overflow-y-auto p-5 space-y-5 min-w-0">
            {/* Score summary */}
            <div className="flex items-baseline gap-3 flex-wrap">
              <span className="text-sm font-semibold text-gray-800">v{selectedVersion}</span>
              {selectedAvg !== null && (
                <span className="text-[13px] font-semibold text-gray-700">{selectedAvg.toFixed(2)} avg</span>
              )}
              <span className="text-[10px] text-gray-400">
                {versionRuns.length} run{versionRuns.length !== 1 ? 's' : ''}
              </span>
              <ScoreDelta current={selectedAvg} previous={prevAvg} prevVersion={prevVersion} />
            </div>

            {/* Prompt diff (collapsible, auto-opens when changes exist) */}
            {selectedVersion && (
              <ChangesSection
                key={selectedVersion}
                selectedVersion={selectedVersion}
                prevVersion={prevVersion}
                promptName={promptName}
              />
            )}

            {/* Runs */}
            <div>
              <div className="text-[11px] font-semibold text-gray-400 uppercase tracking-wide mb-2">
                Runs
              </div>
              {versionRuns.length === 0 ? (
                <p className="text-xs text-gray-400 italic">No runs for this version.</p>
              ) : (
                <div className="space-y-2">
                  {versionRuns.map((run) => <RunDetail key={run.run_id} run={run} />)}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
