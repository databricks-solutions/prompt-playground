import { useState } from 'react';
import { History, Loader2, Maximize2 } from 'lucide-react';
import { useEvalHistory } from '../../hooks/useEvalApi';
import EvalHistoryModal from './EvalHistoryModal';

interface Props {
  promptName: string | null;
  promptVersion: string | null;
  experimentName: string;
}

export default function EvalRunHistory({ promptName, promptVersion, experimentName }: Props) {
  const [showModal, setShowModal] = useState(false);
  const { runs, loading } = useEvalHistory(promptName, experimentName);

  if (!loading && runs.length === 0) return null;

  return (
    <>
      <button
        onClick={() => !loading && setShowModal(true)}
        disabled={loading}
        className="w-full flex items-center gap-2 px-3 py-2 bg-gray-50 border border-gray-100 rounded-lg hover:bg-gray-100 transition-colors text-left disabled:cursor-default"
      >
        <History className="w-3.5 h-3.5 text-gray-400 flex-shrink-0" />
        <span className="text-xs font-semibold text-gray-600 flex-1">
          Eval History
          {!loading && runs.length > 0 && (
            <span className="font-normal text-gray-400"> ({runs.length})</span>
          )}
        </span>
        {loading
          ? <Loader2 className="w-3 h-3 animate-spin text-gray-400 flex-shrink-0" />
          : <Maximize2 className="w-3.5 h-3.5 text-gray-400 flex-shrink-0" />
        }
      </button>

      {showModal && promptName && (
        <EvalHistoryModal
          promptName={promptName}
          currentVersion={promptVersion}
          runs={runs}
          onClose={() => setShowModal(false)}
        />
      )}
    </>
  );
}
