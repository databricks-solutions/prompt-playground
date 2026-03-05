export interface PromptInfo {
  name: string;
  description: string;
  tags: Record<string, string>;
}

export interface PromptVersion {
  version: string;
  description: string;
  aliases: string[];
  template_preview: string;
  creation_timestamp: number | null;
}

export interface PromptTemplate {
  name: string;
  version: string;
  template: string;
  system_prompt: string | null;
  raw_template: string;
  variables: string[];
  tags: Record<string, string>;
  aliases: string[];
}

export interface ModelEndpoint {
  name: string;
  state: string;
  task: string;
}

export interface ScoreDetail {
  name: string;
  value: number | string | null;
  rationale: string | null;
}

export interface EvalRowResult {
  row_index: number;
  variables: Record<string, string>;
  rendered_prompt: string;
  rendered_system_prompt?: string | null;
  response: string;
  score: number | string | null;
  score_rationale: string | null;
  score_details: ScoreDetail[] | null;
}

export interface EvalResponse {
  prompt_name: string;
  prompt_version: string;
  model_name: string;
  dataset: string;
  total_rows: number;
  results: EvalRowResult[];
  avg_score: number | null;
  run_id: string | null;
  experiment_url: string | null;
}

export interface RunResponse {
  rendered_prompt: string;
  system_prompt: string | null;
  response: string;
  model: string;
  usage: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
  run_id?: string | null;
  experiment_url?: string | null;
}

export interface CreatePromptResponse {
  name: string;
  version: string;
  template: string;
  variables: string[];
}

export interface SaveVersionResponse {
  name: string;
  version: string;
  template: string;
  variables: string[];
}

export interface ExperimentInfo {
  name: string;
  experiment_id: string;
  url?: string;
}

export interface JudgeInfo {
  name: string;
  type?: 'custom' | 'guidelines' | 'builtin';
}

export interface EvalTraceRow {
  row_index: number;
  score: number | string | null;
  rationale: string | null;
  details: { name: string; value: number | string | null; rationale: string | null }[] | null;
}

export interface EvalHistoryRun {
  run_id: string;
  run_name: string;
  created_at: number;  // ms epoch
  avg_score: number | null;
  model: string;
  dataset: string;
  scorer: string;
  prompt_version: string;
  total_rows: number | null;
  run_url: string;
}

export interface AppConfig {
  prompt_catalog: string;
  prompt_schema: string;
  eval_catalog: string;
  eval_schema: string;
  mlflow_experiment_name: string;
  sql_warehouse_id: string;
  sql_warehouse_name: string;
}
