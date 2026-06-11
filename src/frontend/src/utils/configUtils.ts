import type { AppConfig } from '../types';

/** Mirrors server `is_app_configured` — prompt catalog/schema always; eval fields when Evaluate tab is on. */
export function isAppConfigured(config: AppConfig | null | undefined): boolean {
  if (!config) return false;
  if (config.is_configured != null) return config.is_configured;
  if (!config.prompt_catalog?.trim() || !config.prompt_schema?.trim()) return false;
  if (!config.evaluate_tab_enabled) return true;
  const evalCatalog = (config.eval_catalog || config.prompt_catalog || '').trim();
  if (!evalCatalog || !config.eval_schema?.trim()) return false;
  if (!config.sql_warehouse_id?.trim()) return false;
  return true;
}
