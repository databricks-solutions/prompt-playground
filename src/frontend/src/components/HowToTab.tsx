import {
  BookOpen,
  ArrowRight,
  Play,
  FlaskConical,
  Database,
  BarChart2,
  GitBranch,
  Layers,
  Cpu,
  CheckCircle2,
  AlertCircle,
  ExternalLink,
  Tag,
  Braces,
  Table2,
  Star,
  ArrowLeftRight,
  FileText,
  Settings,
} from 'lucide-react';

function Section({ icon: Icon, title, color, children }: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  color: string;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
      <div className={`px-5 py-3.5 flex items-center gap-2.5 border-b border-gray-100 ${color}`}>
        <Icon className="w-4 h-4" />
        <h2 className="font-semibold text-sm">{title}</h2>
      </div>
      <div className="px-5 py-4">{children}</div>
    </div>
  );
}

function Step({ num, icon: Icon, title, children }: {
  num: number;
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex gap-3">
      <div className="flex-shrink-0 w-6 h-6 rounded-full bg-databricks-red text-white text-xs font-bold flex items-center justify-center mt-0.5">
        {num}
      </div>
      <div>
        <div className="flex items-center gap-1.5 mb-0.5">
          <Icon className="w-3.5 h-3.5 text-gray-500" />
          <span className="text-sm font-semibold text-gray-800">{title}</span>
        </div>
        <div className="text-xs text-gray-600 leading-relaxed">{children}</div>
      </div>
    </div>
  );
}

function FlowNode({ icon: Icon, label, sub, accent, muted }: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  sub: string;
  accent?: boolean;
  muted?: boolean;
}) {
  return (
    <div className={`flex flex-col items-center gap-1.5 px-4 py-3 rounded-xl border text-center min-w-[120px] ${
      accent
        ? 'bg-databricks-red/5 border-databricks-red/30'
        : muted
        ? 'bg-gray-50 border-dashed border-gray-300'
        : 'bg-gray-50 border-gray-200'
    }`}>
      <Icon className={`w-5 h-5 ${accent ? 'text-databricks-red' : 'text-gray-500'}`} />
      <span className={`text-xs font-semibold ${accent ? 'text-databricks-red' : 'text-gray-700'}`}>{label}</span>
      <span className="text-[10px] text-gray-400 leading-tight">{sub}</span>
    </div>
  );
}

function Tip({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex gap-2 bg-blue-50 border border-blue-100 rounded-lg px-3 py-2 text-xs text-blue-700">
      <AlertCircle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5 text-blue-500" />
      <span>{children}</span>
    </div>
  );
}

function Check({ children }: { children: React.ReactNode }) {
  return (
    <li className="flex gap-2 text-xs text-gray-600">
      <CheckCircle2 className="w-3.5 h-3.5 flex-shrink-0 mt-0.5 text-green-500" />
      <span>{children}</span>
    </li>
  );
}

const QUICK_REFERENCE_ROWS: [string, string, string][] = [
  ['Registered prompts', 'AI/ML → Experiments → [experiment] → Prompts & versions → Prompts', 'Browse versions, aliases, template previews'],
  ['Playground traces', 'AI/ML → Experiments → [experiment] → Observability → Traces', 'Interactive runs logged here with request/response spans'],
  ['Evaluation runs with scores', 'AI/ML → Experiments → [experiment] → Evaluation → Evaluation runs', 'Per-row scores and rationales from the LLM judge'],
  ['Prompt–run linkage', 'AI/ML → Experiments → [experiment] → Prompts & versions → Prompts → [version]', 'Linked Runs tab shows all runs that used a given version'],
  ['Eval datasets', 'Catalog → your catalog → your eval schema', 'Delta tables; any table with matching column names works'],
  ['Model endpoints', 'AI/ML → Serving', 'Filter for READY; Foundation Models listed first'],
];

export default function HowToTab({ evaluateTabEnabled = false }: { evaluateTabEnabled?: boolean }) {
  const quickRefRows = evaluateTabEnabled
    ? QUICK_REFERENCE_ROWS
    : QUICK_REFERENCE_ROWS.filter(([what]) => !what.includes('Evaluation') && what !== 'Eval datasets');

  return (
    <div className="h-full overflow-y-auto bg-gray-50">
      <div className="max-w-4xl mx-auto px-6 py-8 space-y-6">

        {/* Hero */}
        <div className="bg-databricks-dark rounded-xl px-6 py-5 text-white">
          <div className="flex items-start gap-3">
            <div className="w-9 h-9 rounded-lg bg-databricks-red flex items-center justify-center flex-shrink-0 mt-0.5">
              <BookOpen className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-base font-bold mb-1">Prompt Playground — How It Works</h1>
              <p className="text-sm text-gray-300 leading-relaxed">
                The Databricks Prompt Registry stores and versions prompts, but testing them in production
                usually requires writing Python code. This app gives anyone on your team a UI to browse and
                {evaluateTabEnabled ? (
                  <>
                    edit prompts, test them against models, and track runs in Experiments — without writing code.
                    You can also run batch evaluations against datasets from the <strong>Evaluate</strong> tab.
                  </>
                ) : (
                  <>
                    edit prompts and test them against models — without writing code.
                  </>
                )}
              </p>
            </div>
          </div>
        </div>

        {/* Settings */}
        <Section icon={Settings} title="First-Time Setup — App Settings" color="bg-amber-50 text-amber-700">
          <p className="text-xs text-gray-500 mb-4">
            Before using the app, open <strong>Settings</strong> (gear icon in the header) to configure{' '}
            {evaluateTabEnabled
              ? 'where your prompts and evaluation data live'
              : 'which Unity Catalog and schema hold your prompts'}.
            Settings are saved on the server and apply to all users of this app — you only need to do this once.
          </p>
          <div className={`grid grid-cols-1 gap-3 text-xs mb-4 ${evaluateTabEnabled ? 'sm:grid-cols-3' : ''}`}>
            {evaluateTabEnabled && (
              <div className="bg-white rounded-lg border border-amber-100 p-3 space-y-1">
                <p className="font-semibold text-gray-700">SQL Warehouse</p>
                <p className="text-gray-500 leading-relaxed">
                  Required for reading evaluation datasets. Pick any READY warehouse — it will auto-resume if suspended.
                </p>
              </div>
            )}
            <div className="bg-white rounded-lg border border-amber-100 p-3 space-y-1">
              <p className="font-semibold text-gray-700">Prompt Registry</p>
              <p className="text-gray-500 leading-relaxed">The Unity Catalog and schema where your prompts are registered (e.g. <code className="bg-gray-100 px-1 rounded">my_catalog.prompts</code>).</p>
            </div>
            {evaluateTabEnabled && (
              <div className="bg-white rounded-lg border border-amber-100 p-3 space-y-1">
                <p className="font-semibold text-gray-700">Evaluation Data</p>
                <p className="text-gray-500 leading-relaxed">The catalog and schema where your eval datasets (Delta tables) live. Can be the same catalog as prompts or a different one.</p>
              </div>
            )}
          </div>
          <Tip>
            The app opens Settings automatically on first load if it hasn't been configured yet.
            After saving, the prompt list reloads immediately from the new location.
          </Tip>
        </Section>

        {/* Tab overview */}
        <Section
          icon={ArrowLeftRight}
          title={evaluateTabEnabled ? 'How the Three Tabs Work Together' : 'How the Two Tabs Work Together'}
          color="bg-indigo-50 text-indigo-700"
        >
          <p className="text-xs text-gray-500 mb-4">
            {evaluateTabEnabled ? (
              <>
                The natural workflow is: browse and pick a prompt in <strong>Prompts</strong>, test it interactively in{' '}
                <strong>Playground</strong>, then run a batch evaluation in <strong>Evaluate</strong>.
              </>
            ) : (
              <>
                The natural workflow is: browse and pick a prompt in <strong>Prompts</strong>, then test it interactively in{' '}
                <strong>Playground</strong>.
              </>
            )}
          </p>
          <div className="flex flex-wrap items-center gap-2 justify-center mb-5">
            <div className="flex flex-col items-center gap-1.5 px-4 py-3 rounded-xl border text-center bg-gray-50 border-gray-200">
              <FileText className="w-5 h-5 text-gray-500" />
              <span className="text-xs font-semibold text-gray-700">Prompts</span>
              <span className="text-[10px] text-gray-400 leading-tight">Browse registry<br/>Pick prompt + version</span>
            </div>
            <ArrowRight className="w-4 h-4 text-gray-400 flex-shrink-0" />
            <div className="flex flex-col items-center gap-1.5 px-4 py-3 rounded-xl border text-center bg-red-50 border-databricks-red/30">
              <Play className="w-5 h-5 text-databricks-red" />
              <span className="text-xs font-semibold text-databricks-red">Playground</span>
              <span className="text-[10px] text-gray-400 leading-tight">Fill variables<br/>Test with sample inputs</span>
            </div>
            {evaluateTabEnabled && (
              <>
                <ArrowRight className="w-4 h-4 text-gray-400 flex-shrink-0" />
                <div className="flex flex-col items-center gap-1.5 px-4 py-3 rounded-xl border text-center bg-purple-50 border-purple-300">
                  <FlaskConical className="w-5 h-5 text-purple-600" />
                  <span className="text-xs font-semibold text-purple-700">Evaluate</span>
                  <span className="text-[10px] text-gray-400 leading-tight">Run against a full dataset</span>
                </div>
              </>
            )}
            <ArrowRight className="w-4 h-4 text-gray-400 flex-shrink-0" />
            <FlowNode icon={BarChart2} label="Experiments" sub="In Databricks UI" muted />
          </div>
          <p className="text-[11px] text-gray-400 text-center mb-4">
            {evaluateTabEnabled ? (
              <>
                Playground and evaluation runs are automatically logged to the <strong>Experiments</strong> section of
                Databricks — separate from this app, where you can compare versions and review traces over time.
              </>
            ) : (
              <>
                Playground runs are automatically logged to the <strong>Experiments</strong> section of Databricks —
                separate from this app, where you can compare versions and review traces over time.
              </>
            )}
          </p>
        </Section>

        {/* Architecture */}
        <Section icon={Layers} title="How This App Connects to Databricks" color="bg-gray-50 text-gray-700">
          <div className="flex flex-wrap items-center gap-2 justify-center mb-5">
            <FlowNode icon={GitBranch} label="Prompt Registry" sub="MLflow / UC" />
            <ArrowRight className="w-4 h-4 text-gray-400 flex-shrink-0" />
            <FlowNode icon={Play} label="Playground" sub="This app" accent />
            <ArrowRight className="w-4 h-4 text-gray-400 flex-shrink-0" />
            <FlowNode icon={Cpu} label="Model Serving" sub="AI Gateway" />
            <ArrowRight className="w-4 h-4 text-gray-400 flex-shrink-0" />
            <FlowNode icon={BarChart2} label="Experiments" sub="MLflow tracking" />
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
            <div className="bg-gray-50 rounded-lg p-3 space-y-1.5">
              <div className="flex items-center gap-1.5 font-semibold text-gray-700">
                <GitBranch className="w-3.5 h-3.5" /> MLflow Prompt Registry
              </div>
              <p className="text-gray-500 leading-relaxed">
                Source of truth for prompts. Engineers register versions with{' '}
                <code className="bg-white border border-gray-200 px-1 rounded">{'{{variable}}'}</code>{' '}
                placeholders and aliases like <code className="bg-white border border-gray-200 px-1 rounded">production</code>.
                Stored in Unity Catalog — configure which catalog and schema to use in <strong>Settings</strong> (gear icon).
              </p>
            </div>
            <div className="bg-gray-50 rounded-lg p-3 space-y-1.5">
              <div className="flex items-center gap-1.5 font-semibold text-gray-700">
                <Cpu className="w-3.5 h-3.5" /> AI Gateway / Model Serving
              </div>
              <p className="text-gray-500 leading-relaxed">
                Databricks Foundation Model API endpoints and any custom serving endpoints.
                The model selector lists every READY chat endpoint — Foundation Models appear first.
              </p>
            </div>
            <div className="bg-gray-50 rounded-lg p-3 space-y-1.5">
              <div className="flex items-center gap-1.5 font-semibold text-gray-700">
                <BarChart2 className="w-3.5 h-3.5" /> Experiments (Databricks UI)
              </div>
              <p className="text-gray-500 leading-relaxed">
                {evaluateTabEnabled ? (
                  <>
                    Every run — playground and evaluation — is automatically logged to a GenAI experiment and linked to
                    the specific prompt version. Find it under <strong>AI/ML → Experiments</strong> in Databricks.
                  </>
                ) : (
                  <>
                    Every playground run is automatically logged to a GenAI experiment and linked to the specific prompt
                    version. Find it under <strong>AI/ML → Experiments</strong> in Databricks.
                  </>
                )}
              </p>
            </div>
            {evaluateTabEnabled && (
              <div className="bg-gray-50 rounded-lg p-3 space-y-1.5">
                <div className="flex items-center gap-1.5 font-semibold text-gray-700">
                  <Database className="w-3.5 h-3.5" /> Unity Catalog (Eval Data)
                </div>
                <p className="text-gray-500 leading-relaxed">
                  Eval datasets are Delta tables in Unity Catalog. The Evaluate tab reads rows via SQL Warehouse and maps
                  columns to prompt variables.
                </p>
              </div>
            )}
          </div>
        </Section>

        {/* Prompts tab */}
        <Section icon={FileText} title="Prompts Tab — Browse, Manage, Edit" color="bg-gray-50 text-gray-700">
          <div className="space-y-4">
            <Step num={1} icon={GitBranch} title="Browse Your Prompt Registry">
              Your prompt registry location is configured in <strong>Settings</strong> (gear icon in the header).
              The app loads all prompts from that catalog and schema automatically. The{' '}
              <strong>Experiment</strong> selector in the header picks which MLflow experiment runs are
              logged to — search by name (e.g. <em>hinge</em>), no path typing required. The app also
              suggests experiments linked to your prompt catalog when possible. Check{' '}
              <strong>Filter prompts to experiment</strong> (next to the selector) to show only prompts
              that have been run in that experiment — the count updates automatically. This
              filter applies across the Prompts tab{evaluateTabEnabled ? ' and Evaluate tab' : ''}.
            </Step>
            <Step num={2} icon={Tag} title="Browse Prompts and Versions">
              Select a prompt from the searchable dropdown and click a version to select it. Each version
              card shows a template preview, description, and any aliases (like{' '}
              <code className="bg-gray-100 px-1 rounded">production</code>). The <strong>Prompt Preview</strong>{' '}
              panel shows the full template with variables highlighted — click <strong>Raw</strong> to see
              the template with <code className="bg-gray-100 px-1 rounded">&lt;system&gt;</code>/<code className="bg-gray-100 px-1 rounded">&lt;user&gt;</code> tags included.
            </Step>
            <Step num={3} icon={Braces} title="Edit or Create Prompts">
              Click <strong>New Version</strong> in the Preview panel header to edit the template in a
              textarea. Saving registers a new version in the Prompt Registry — the original is never changed.
              Click the <strong>+</strong> icon next to the Prompt label to create a brand-new prompt from scratch.
            </Step>
            <Step num={4} icon={Play} title="Test in Playground">
              Once you've selected a prompt and version, click <strong>Test in Playground →</strong> at
              the bottom of the left panel.
            </Step>
            <Tip>
              <strong>No prompt yet?</strong> Click the <strong>+</strong> icon next to "Prompt" to register
              a new prompt — fill in a name, optional description, and template.
            </Tip>
          </div>
        </Section>

        {/* Playground tab */}
        <Section icon={Play} title="Playground Tab — Test Interactively" color="bg-red-50 text-databricks-red">
          <div className="space-y-4">
            <Step num={1} icon={Tag} title="Load a Prompt">
              The loaded prompt name and version are shown at the top of the left panel. If nothing is
              selected yet, click <strong>Select a prompt →</strong> to go to the Prompts tab. Click{' '}
              <strong>Change →</strong> to switch.
            </Step>
            <Step num={2} icon={Braces} title="Fill in Template Variables">
              Input fields appear for every{' '}
              <code className="bg-gray-100 px-1 rounded">{'{{variable}}'}</code> in the template.
              The <strong>Prompt Preview</strong> panel updates live as you type, with unfilled variables
              highlighted in amber.
            </Step>
            <Step num={3} icon={Cpu} title="Select a Model and Run">
              Choose any READY serving endpoint from the model dropdown. Adjust Max Tokens and Temperature
              in <strong>Run Settings</strong> if needed, then click <strong>Run Prompt</strong>. The response
              appears in the bottom panel with token usage and an <strong>Open in Databricks ↗</strong> link.
            </Step>
          </div>
        </Section>

        {evaluateTabEnabled && (
          <Section icon={FlaskConical} title="Evaluate Tab — Batch Evaluation" color="bg-purple-50 text-purple-700">
            <div className="space-y-4">
              <Step num={1} icon={Tag} title="Select Prompt and Version">
                Pick a prompt and version from the searchable dropdowns. A template preview appears inline
                with variable placeholders highlighted.
              </Step>
              <Step num={2} icon={Table2} title="Pick an Eval Dataset and Map Variables">
                Set the eval dataset catalog and schema, pick a table, and set <strong>Max Rows</strong> (1–20).
                Map each <code className="bg-gray-100 px-1 rounded">{'{{variable}}'}</code> to a column —
                use <strong>Auto</strong> to fill exact-name matches automatically.
              </Step>
              <Step num={3} icon={Cpu} title="Select a Model">
                Pick the model endpoint to call for each row. Foundation Models appear first; all
                <strong> READY</strong> serving endpoints are listed.
              </Step>
              <Step num={4} icon={Star} title="Configure the Judge">
                <ul className="mt-1 space-y-1 list-none">
                  <li><strong>Default quality scorer</strong> — built-in 1–5 LLM-as-judge (helpfulness, accuracy, completeness). Uses your selected model by default; check <em>Use a different model for judging</em> to override. Adjust temperature if needed.</li>
                  <li className="mt-1"><strong>Built-in presets</strong> — Safety, Relevance, Fluency, Completeness, Summarization, Correctness. No config needed.</li>
                  <li className="mt-1"><strong>Registered judges</strong> — reusable, saved per-experiment. Click <strong>+ New</strong> to create a <em>Custom</em> (free-form instructions) or <em>Guidelines</em> (rule checklist with per-rule pass/fail) judge.</li>
                </ul>
              </Step>
              <Step num={5} icon={Play} title="Run and Review Results">
                Click <strong>Run Evaluation</strong>. A summary banner shows the average score, metadata,
                and MLflow Run ID. Use <strong>Open in Databricks</strong> to jump to the run in Experiments.
                <br /><br />
                Low-scoring rows are highlighted in red. Click the <strong>Score</strong> header to sort.
                Expand any row to see the rendered prompt, full response, and judge output.
              </Step>
            </div>
          </Section>
        )}

        {/* What you see in Databricks */}
        <Section icon={BarChart2} title="What to Look For in Databricks" color="bg-green-50 text-green-700">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <p className="text-xs font-semibold text-gray-700 mb-2 flex items-center gap-1.5">
                <BarChart2 className="w-3.5 h-3.5" /> Experiments
              </p>
              <ul className="space-y-1.5">
                <Check>Under <strong>Observability → Traces</strong>: playground runs logged here, tagged with prompt name and version</Check>
                {evaluateTabEnabled && (
                  <Check>Under <strong>Evaluation → Evaluation runs</strong>: batch eval runs with avg_score metric and per-row scores</Check>
                )}
                <Check>Under <strong>Prompts & versions → Prompts</strong>: the prompt registry — browse versions, see linked runs</Check>
                <Check>Click any trace to see the full request, response{evaluateTabEnabled ? ', score, and rationale' : ', and model output'}</Check>
              </ul>
            </div>
            <div>
              <p className="text-xs font-semibold text-gray-700 mb-2 flex items-center gap-1.5">
                <GitBranch className="w-3.5 h-3.5" /> Prompt Registry
              </p>
              <ul className="space-y-1.5">
                <Check>Navigate to <strong>AI/ML → Experiments → [your experiment] → Prompts & versions → Prompts</strong></Check>
                <Check>Click a version — the <strong>Linked Runs</strong> tab shows every {evaluateTabEnabled ? 'playground and eval run' : 'playground run'} that used it</Check>
                <Check>
                  Full audit trail: which prompt version, which model
                  {evaluateTabEnabled ? ', and what scores' : ', and what was sent and returned'}
                </Check>
                <Check>Compare v1 vs v2 runs side by side to decide which version to promote to <strong>production</strong></Check>
              </ul>
            </div>
          </div>
        </Section>

        {/* Quick reference */}
        <Section icon={ExternalLink} title="Quick Reference — Where Things Live in Databricks" color="bg-gray-50 text-gray-700">
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-gray-500 border-b border-gray-100">
                  <th className="text-left pb-2 font-semibold">What</th>
                  <th className="text-left pb-2 font-semibold">Where in Databricks</th>
                  <th className="text-left pb-2 font-semibold">Notes</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {quickRefRows.map(([what, where, notes]) => (
                  <tr key={what} className="text-gray-600">
                    <td className="py-1.5 pr-4 font-medium text-gray-800 whitespace-nowrap">{what}</td>
                    <td className="py-1.5 pr-4"><code className="bg-gray-100 px-1 rounded text-[11px]">{where}</code></td>
                    <td className="py-1.5 text-gray-500">{notes}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>

      </div>
    </div>
  );
}
