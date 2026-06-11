import type { ExperimentInfo } from '../types';

/** Short label for dropdowns (last path segment). Full path in description. */
export function experimentSelectOptions(experiments: Array<{ name: string }>) {
  return experiments.map((e) => {
    const short = e.name.split('/').filter(Boolean).pop() ?? e.name;
    return {
      value: e.name,
      label: short,
      description: short !== e.name ? e.name : undefined,
    };
  });
}

export function pickSuggestedExperiment(
  candidates: Array<Pick<ExperimentInfo, 'name'>>,
  catalog: string,
): Pick<ExperimentInfo, 'name'> | null {
  if (candidates.length === 0) return null;
  if (candidates.length === 1) return candidates[0];
  const slug = catalog.trim().toLowerCase().replace(/_/g, '-');
  return (
    candidates.find((e) => e.name.toLowerCase().includes('prompt-playground')) ??
    candidates.find((e) => slug && e.name.toLowerCase().includes(slug)) ??
    candidates[0]
  );
}

export const PP_SELECTED_EXPERIMENT_KEY = 'pp-selected-experiment';

export function readStoredExperimentName(): string {
  try {
    return sessionStorage.getItem(PP_SELECTED_EXPERIMENT_KEY)?.trim() ?? '';
  } catch {
    return '';
  }
}

export function writeStoredExperimentName(name: string) {
  try {
    if (name.trim()) sessionStorage.setItem(PP_SELECTED_EXPERIMENT_KEY, name.trim());
    else sessionStorage.removeItem(PP_SELECTED_EXPERIMENT_KEY);
  } catch {
    /* ignore */
  }
}
