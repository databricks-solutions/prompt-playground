import { highlightVarsInDiff, type DiffLine } from '../utils/diffUtils';

// DiffLines: renders lines without an outer wrapper — use inside a container that provides border/font
export function DiffLines({ lines }: { lines: DiffLine[] }) {
  return (
    <>
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
              dangerouslySetInnerHTML={{ __html: highlightVarsInDiff(line.text) }}
            />
          </div>
        );
      })}
    </>
  );
}

// DiffView: standalone with border, for raw-mode and eval history contexts
export default function DiffView({ lines }: { lines: DiffLine[] }) {
  return (
    <div className="rounded-lg border border-gray-200 overflow-hidden font-mono text-[11px] leading-relaxed">
      <DiffLines lines={lines} />
    </div>
  );
}
