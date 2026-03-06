import type { PromptTemplate } from '../types';

export type DiffLine = { type: 'added' | 'removed' | 'unchanged'; text: string };

export function computeLineDiff(oldText: string, newText: string): DiffLine[] {
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

export function getTemplateText(tpl: PromptTemplate): string {
  return (tpl.raw_template ?? tpl.template).replace(/\\n/g, '\n');
}

export function highlightVarsInDiff(text: string): string {
  const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  return esc(text).replace(
    /\{\{\s*(\w+)\s*\}\}/g,
    (_, key) => `<span class="inline-block bg-purple-100 text-purple-700 rounded px-0.5 font-mono">{{${key}}}</span>`
  );
}
