"""MLflow GenAI evaluation dataset SDK wrapper.

Provides experiment-scoped dataset discovery, schema introspection, and record
reading using the mlflow.genai.datasets API — replacing the previous raw-SQL
approach that read flat UC tables via SQL Warehouse.

MLflow GenAI eval datasets use a structured schema:
  - inputs: dict[str, Any]        (required — app inputs like user question, context)
  - expectations: dict[str, Any]  (optional — ground truth with reserved keys)
  - dataset_record_id, create_time, created_by, etc. (auto-managed metadata)
"""

import logging
from typing import Any

import mlflow
from server.mlflow_helpers import configure_mlflow

logger = logging.getLogger(__name__)


def list_datasets(experiment_name: str) -> list[dict[str, str]]:
    """List eval datasets associated with an experiment.

    Returns a list of dicts with 'name' and 'dataset_id' keys.
    """
    configure_mlflow()
    exp = mlflow.get_experiment_by_name(experiment_name)
    if not exp:
        return []

    from mlflow.genai.datasets import search_datasets

    datasets = search_datasets(experiment_ids=exp.experiment_id)
    return [
        {"name": ds.name, "dataset_id": getattr(ds, "dataset_id", ds.name)}
        for ds in datasets
    ]


def get_dataset_schema(dataset_name: str) -> dict[str, list[str]]:
    """Return the keys found inside `inputs` and `expectations` dicts.

    Reads a sample of records and extracts the union of keys across rows.
    Returns {"input_keys": [...], "expectation_keys": [...]}.
    """
    configure_mlflow()
    from mlflow.genai.datasets import get_dataset

    ds = get_dataset(name=dataset_name)
    df = ds.to_df()

    input_keys: set[str] = set()
    expectation_keys: set[str] = set()

    for _, row in df.head(50).iterrows():
        inputs = row.get("inputs")
        if isinstance(inputs, dict):
            input_keys.update(inputs.keys())
        expectations = row.get("expectations")
        if isinstance(expectations, dict):
            expectation_keys.update(expectations.keys())

    return {
        "input_keys": sorted(input_keys),
        "expectation_keys": sorted(expectation_keys),
    }


def read_dataset_records(
    dataset_name: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Read records from an eval dataset as a list of dicts.

    Each dict has 'inputs' (dict), 'expectations' (dict or None),
    and optionally 'dataset_record_id'.
    """
    configure_mlflow()
    from mlflow.genai.datasets import get_dataset

    ds = get_dataset(name=dataset_name)
    df = ds.to_df()

    if limit and limit < len(df):
        df = df.head(limit)

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        record: dict[str, Any] = {
            "inputs": row.get("inputs") if isinstance(row.get("inputs"), dict) else {},
        }
        expectations = row.get("expectations")
        if isinstance(expectations, dict) and expectations:
            record["expectations"] = expectations
        record_id = row.get("dataset_record_id")
        if record_id:
            record["dataset_record_id"] = str(record_id)
        records.append(record)

    return records


def count_dataset_records(dataset_name: str) -> int:
    """Return the total number of records in an eval dataset."""
    configure_mlflow()
    from mlflow.genai.datasets import get_dataset

    ds = get_dataset(name=dataset_name)
    df = ds.to_df()
    return len(df)
