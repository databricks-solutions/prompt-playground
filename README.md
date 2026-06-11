# Prompt Playground

Prompt Playground is an interactive, no-code Databricks App for designing and testing prompts stored in the [Prompt Registry](https://docs.databricks.com/aws/en/mlflow3/genai/prompt-version-mgmt/prompt-registry/). It enables product owners, prompt engineers, and both technical and non-technical users to iterate on prompt templates and run them against live model serving endpoints — without writing code.

- **Manage prompts** — browse, create, and version prompt templates directly from the UI
- **Iterate interactively** — fill in `{{template_variables}}`, run against any model serving endpoint, and preview the fully rendered prompt before executing
- **Tightly integrated with Databricks** — playground runs are logged as MLflow traces with direct links to the Experiments UI; all data stays in your Unity Catalog environment
- **Batch evaluation (optional)** — enable the experimental **Evaluate** tab in Settings to run prompts against Unity Catalog Delta tables with LLM-as-judge scoring (off by default)

## Installation

### Prerequisites

- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/install.html) `>= 0.220.0`, authenticated via `databricks auth login`
- A **model serving endpoint** — [Foundation Model API](https://docs.databricks.com/machine-learning/foundation-models/index.html) endpoints work out of the box
- The app's **service principal** must have `USE CATALOG` / `USE SCHEMA` and `MANAGE` (or appropriate read/write) on the Unity Catalog schema where your prompts are registered

**Additional prerequisites if you enable the Evaluate tab:**

- A **SQL Warehouse** (used to read eval datasets)
- `MANAGE` (or read access) on the schema where evaluation datasets live

### Bundle configuration

Bundle variables ship **empty by default** — each workspace admin picks catalog, schema, experiment, and (if needed) SQL warehouse in the in-app **Settings** panel after deploy. No SQL warehouse is required at deploy time.

```yaml
variables:
  prompt_catalog:
    default: ""
  prompt_schema:
    default: ""
  evaluate_tab_enabled:
    default: "false" # set "true" to show the experimental Evaluate tab
  eval_catalog:
    default: ""
  eval_schema:
    default: ""
```

Set **MLflow experiment** and **SQL warehouse** in Settings after deploy — they are not bundle variables.

### Service principal permissions

Grant the app's service principal (see **Compute → Apps → your app → Identity**) at minimum:

| Resource | Privilege | Required for |
|----------|-----------|--------------|
| Prompt catalog | `USE CATALOG` | Prompts + Playground |
| Prompt schema | `USE SCHEMA`, `CREATE FUNCTION`, `EXECUTE`, `MANAGE` | Browse/create prompts (`MANAGE` is not included in `ALL PRIVILEGES`) |
| MLflow experiment | `CAN MANAGE` or create experiment | Playground traces |
| Model serving endpoint | `CAN QUERY` (via bundle) | Running prompts |
| Eval dataset catalog/schema | `USE CATALOG`, `USE SCHEMA`, `SELECT` on tables | Evaluate tab only |
| SQL warehouse | `CAN USE` | Evaluate tab only (pick in Settings — not configured in the bundle) |

Example SQL grants:

```sql
GRANT USE CATALOG ON CATALOG my_catalog TO `<service-principal-client-id>`;
GRANT USE SCHEMA, CREATE FUNCTION, EXECUTE, MANAGE ON SCHEMA my_catalog.prompts TO `<service-principal-client-id>`;
GRANT SELECT ON SCHEMA my_catalog.eval_data TO `<service-principal-client-id>`;
```

To hide legacy foundation models from the model dropdown, set `DEPRECATED_MODEL_ENDPOINTS` (comma-separated endpoint names) on the app.

### Setup

**1. Clone the repository**

```bash
git clone https://github.com/databricks-solutions/prompt-playground.git
cd prompt-playground
```

**2. Deploy**

```bash
databricks bundle validate
databricks bundle deploy
databricks bundle run prompt_playground
```

The app URL will be printed in the output. You can also find it under **Compute > Apps** in your workspace.

## Usage

> **First time?** Open the **How to Use** tab for a walkthrough. Click **Settings** (gear icon) and set your **Prompt Registry** catalog and schema — that's all you need for Prompts and Playground. Enable **Show Evaluate tab** in Settings only if you want batch evaluation.

**Register your first prompt from within the app:**

1. Open the Prompt Playground app
2. Click the **+** icon next to the Prompt selector
3. Fill in a name, optional description, and template (use `{{variable}}` placeholders)
4. Click **Create Prompt** — the new prompt is registered and immediately selected

## Troubleshooting

**App fails to start / "MANAGE privilege" error**
→ The service principal is missing required privileges on the prompts schema.

**"No prompts found"**
→ Open Settings and verify your prompt catalog and schema. Confirm the app's service principal has access to that schema.

**Eval datasets not loading**
→ The Evaluate tab must be enabled in Settings. Verify eval catalog/schema and SQL warehouse, and confirm the service principal can read those tables.

**Model endpoint not listed**
→ The endpoint may not be in `READY` state. Check **Serving > Endpoints** in your workspace.

**Experiment dropdown is slow or empty**
→ Set **MLflow experiment** in Settings (or leave blank and pick from the capped workspace browse list).

## How to get help

For questions or bugs, please contact agents-outreach@databricks.com and the team will reach out shortly.

## License

This project is licensed under the [Databricks DB License](LICENSE.md).

| library | description | license | source |
|---------|-------------|---------|--------|
| React | Frontend framework | MIT | https://github.com/facebook/react |
| FastAPI | Backend web framework | MIT | https://github.com/tiangolo/fastapi |
| Tailwind CSS | Utility-first CSS | MIT | https://github.com/tailwindlabs/tailwindcss |
| Vite | Frontend build tool | MIT | https://github.com/vitejs/vite |
| MLflow | ML lifecycle management | Apache 2.0 | https://github.com/mlflow/mlflow |
| Lucide React | Icon library | ISC | https://github.com/lucide-icons/lucide |
