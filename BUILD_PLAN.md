# Build plan: OpenAI-style prompts & non-deprecated model list

Targets: **Prompt Playground** (`feat/eval-reliability` / main worktree).  
Audience: engineering; refine estimates and “definition of done” before implementation.

---

## Current state (baseline)

| Area | Today |
|------|--------|
| **Templates** | `{{variable}}` Jinja-style placeholders; optional `<system>` / `<user>` XML blocks in a single string for role separation (`src/server/templates.py`, `src/frontend/src/utils/templateUtils.ts`). |
| **MLflow / UC** | `mlflow.genai.load_prompt` returns `template` as **str or `list[dict]`** with `role` / `content`. Backend already maps native message lists to XML for editor round-trip (`mlflow_client.get_prompt_template`). |
| **Model calls** | OpenAI-compatible SDK: `messages=[{role, content}, …]` (`server/llm.py` `call_model`). |
| **Model list** | `list_serving_endpoints()` filters by task, excludes name patterns and embeddings (`server/llm.py`). No explicit “deprecated” filter. |

---

## Initiative A — Prompts in OpenAI formatting

### Goal (refine with PM)

Pick one primary meaning (they can be phased):

1. **Storage / registry** — Create and persist prompt versions as **chat messages** (`[{ "role": "system"|"user"|…, "content": "…" }]`) in the Prompt Registry, with variables still expressed consistently (e.g. `{{name}}` inside `content` strings), **or** adopt MLflow’s documented template format if it differs.
2. **Authoring UX** — Replace (or supplement) the XML-in-one-textarea pattern with an **OpenAI-inspired editor**: separate system / user / assistant blocks, or a small message list with add/remove rows, synced to the registry format on save.
3. **Compatibility** — Existing XML-based prompts continue to load; optional one-time or lazy **migration** to message-list form.

### Technical work (draft backlog)

1. **Product / UX**
   - Wireframes for single vs multi-message prompts; error states; how eval column mapping applies to which message.
   - Decide variable scope: all `{{vars}}` across messages vs user-only.

2. **Backend**
   - Extend `get_prompt_template` / `create_prompt_version` flows so **round-trip** preserves OpenAI-style lists without forcing lossy XML (today list → XML is one-way for fidelity in the editor).
   - Unify `parse_system_user`, `parse_template_variables`, and `render_template` (or add `render_messages`) so eval + `/run` construct the same `messages` payload OpenAI expects.
   - **Migration helper** (optional): function that converts legacy XML string → `list[dict]` for write-back.

3. **Frontend**
   - `PromptForm`, `PromptPreview`, `EvaluatePanel`, `DatasetTable`: consume message-list model or a stable DTO from `/api`.
   - Update placeholders / help text (today references XML in `PromptForm.tsx`, `PromptPreview.tsx`).
   - Keep `parseSystemUser` / `buildXmlTemplate` paths until legacy prompts are migrated or explicitly dual-mode.

4. **Tests**
   - `templates.py` and `mlflow_client` round-trip tests for list templates with variables in system and user messages.
   - API tests for create/update/load prompt versions.

### Risks

- MLflow Prompt Registry constraints on `template` type/versioning — validate with UC + MLflow 3 docs before locking schema.
- Eval UX: column mapping today assumes rendered strings; multi-message may need clearer labels (“user message column”).

### Acceptance criteria (starter)

- [ ] New prompts can be authored and saved in the chosen OpenAI / message-list format and appear correctly in the Traces / registry.
- [ ] `/api/run` and batch eval send the intended `messages` to serving.
- [ ] Legacy XML templates still load and run until migrated.

---

## Initiative B — Hide deprecated models in the model list

### Goal

The model dropdown should **omit endpoints that are deprecated** (Databricks-defined or workspace policy), without hiding valid GA models.

### Discovery (required first)

1. **Inspect live API payloads** in a real workspace: `ServingEndpoint` / `ServingEndpointDetailed` from `w.serving_endpoints.list()` and `get(name)` for a known deprecated vs current foundation endpoint.
2. Check for fields such as: tags, `config`, `served_entities`, task type, documentation URLs, or **billing / lifecycle** markers (naming varies by release).
3. If **no stable API field** exists, fall back to:
   - **Configurable blocklist** in `pp_settings.json` or env (e.g. `DEPRECATED_MODEL_ENDPOINTS` or prefix list), **and/or**
   - **Allowlist** of `databricks-*` foundation names from public docs, updated periodically (heavier maintenance).

### Technical work (draft backlog)

1. **`list_serving_endpoints()`** (`server/llm.py`)
   - After listing (or via batched `get` if list is shallow), **filter** deprecated endpoints using discovered criteria.
   - Keep existing filters (embeddings, internal name patterns) unless they subsume new logic.
2. **Settings**
   - Optional admin-tunable blocklist / “hide legacy serving” flag so customers can tune without a deploy.
3. **Tests**
   - Mock endpoint objects with/without deprecation signals; assert filtered list.
4. **Docs**
   - Short note in README or troubleshooting: how to override blocklist if a needed model is wrongly hidden.

### Risks

- **False negatives**: hiding a model users still rely on → mitigate with settings override.
- **Performance**: per-endpoint `get()` if list doesn’t include metadata — cache or batch.

### Acceptance criteria (starter)

- [ ] Deprecated foundation / legacy endpoints documented by Databricks for your workspace **do not** appear in `/api/models` by default.
- [ ] Clear override path (config blocklist / env) documented.
- [ ] Unit tests cover filter behavior with mocked SDK responses.

---

## Suggested order

1. **Spike B (discovery)** — 0.5–1 day: capture real SDK shapes for deprecated endpoints; decide filter vs blocklist.
2. **A (UX + schema decision)** — align on message-list vs XML+compat before large frontend work.
3. **Implement A backend** — registry + render path.
4. **Implement A frontend** — editor + preview.
5. **Implement B** — filter + settings + tests.
6. **Regression / migration** — legacy prompts, eval flows, smoke in workspace.

---

## Out of scope (unless you pull them in)

- Changing OpenAI SDK or serving URL semantics (already OpenAI-compatible).
- Automatic migration of all existing UC prompts without an explicit migration story.
