"""Seed Hinge-style demo prompts into the UC Prompt Registry.

Profile-prompt-centric set with minimal user input (small vars like `tone`,
never "paste your whole profile"). Removes the earlier broad/over-input set
first, then creates the current 5 prompts (a few with multiple versions +
a `production` alias) under hinge_prompt_app.prompts.

Run:  source .venv/bin/activate && python scripts/seed_hinge_prompts.py

Safe to re-run: deletes the listed prompts first, then recreates.
"""

import os
import mlflow
from mlflow import MlflowClient

PROFILE = os.environ.get("DATABRICKS_PROFILE", "e2-demo-field-eng")
CATALOG = "hinge_prompt_app"
SCHEMA = "prompts"
EXPERIMENT = "/Shared/hinge-prompt-playground"

mlflow.set_tracking_uri(f"databricks://{PROFILE}")
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()


def fq(name: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{name}"


# Every prompt we manage in this schema — deleted before reseeding so the demo
# set is exactly the list below (no leftovers from earlier iterations).
ALL_NAMES = [
    # removed in earlier iterations
    "bio_polish", "opening_line", "match_rationale", "prompt_answer", "date_idea",
    "icebreaker_from_prompt",
    # current set
    "profile_answer_coach", "prompt_picker", "convo_reviver", "first_date_idea",
]

HINGE_PROMPT_MENU = (
    "• Two truths and a lie\n"
    "• My simple pleasures\n"
    "• Together, we could...\n"
    "• I go crazy for...\n"
    "• Dating me is like...\n"
    "• The way to win me over is...\n"
    "• A shower thought I recently had...\n"
    "• My most irrational fear...\n"
    "• I'll know it's time to delete Hinge when...\n"
    "• Green flags I look for...\n"
    "• Unusual skills...\n"
    "• The key to my heart is..."
)

COACH_EXAMPLES = (
    "Examples of strong answers:\n"
    "• \"Two truths and a lie\" → \"I've eaten guinea pig in Peru, I can land a "
    "kickflip, and I've read every Brontë novel.\" (one's a lie \U0001f609)\n"
    "• \"My simple pleasures\" → \"The first sip of coffee before the house wakes up, "
    "and beating my dad at chess (currently down 0–47).\"\n"
    "• \"The way to win me over is\" → \"Take me to the weirdest restaurant you know "
    "and let me order for both of us.\""
)

# Each prompt: list of versions (template + version-level description).
# `alias` maps an alias name -> version index (1-based).
PROMPTS = [
    {
        "name": "profile_answer_coach",
        "description": "Rewrite a member's answer to a Hinge profile prompt so it's sharper and more you.",
        "alias": {"production": 3},
        "versions": [
            {
                "description": "Initial version — simple rewrite.",
                "template": (
                    "<system>\n"
                    "You are a Hinge profile coach. Rewrite the member's answer to a Hinge profile "
                    "prompt so it's more specific and engaging, while keeping their authentic voice.\n"
                    "</system>\n\n"
                    "<user>\n"
                    "Hinge prompt: \"{{prompt_question}}\"\n"
                    "My current answer: \"{{current_answer}}\"\n\n"
                    "Rewrite it in a {{tone}} tone.\n"
                    "</user>"
                ),
            },
            {
                "description": "Adds worked examples, a no-cliches guardrail, and a length cap.",
                "template": (
                    "<system>\n"
                    "You are a Hinge profile coach. Rewrite the member's answer to a Hinge profile "
                    "prompt so it's specific, intriguing, and keeps their authentic voice. Avoid "
                    "cliches, humble-brags, and anything anyone could say.\n\n"
                    f"{COACH_EXAMPLES}\n"
                    "</system>\n\n"
                    "<user>\n"
                    "Hinge prompt: \"{{prompt_question}}\"\n"
                    "My current answer: \"{{current_answer}}\"\n\n"
                    "Rewrite it in a {{tone}} tone. Keep it under 30 words.\n"
                    "</user>"
                ),
            },
            {
                "description": "Requires a closing hook and a one-line note on what changed.",
                "template": (
                    "<system>\n"
                    "You are a Hinge profile coach. You help members rewrite their answers to Hinge "
                    "profile prompts so they're specific, intriguing, and easy to reply to — while "
                    "keeping the member's authentic voice.\n\n"
                    "Great Hinge answers are concrete, show rather than tell, and leave an obvious "
                    "hook to comment on. Avoid cliches, humble-brags, and anything anyone could say.\n\n"
                    f"{COACH_EXAMPLES}\n"
                    "</system>\n\n"
                    "<user>\n"
                    "Hinge prompt: \"{{prompt_question}}\"\n"
                    "My current answer: \"{{current_answer}}\"\n\n"
                    "Rewrite my answer in a {{tone}} tone. Keep it under 30 words and end with a clear "
                    "hook. Then add one short line on what you changed and why.\n"
                    "</user>"
                ),
            },
        ],
    },
    {
        "name": "prompt_picker",
        "description": "Recommend which Hinge profile prompts a member should answer to stand out.",
        "alias": {"production": 1},
        "versions": [
            {
                "description": "Initial version.",
                "template": (
                    "<system>\n"
                    "You are a Hinge profile coach. You help members choose which Hinge profile "
                    "prompts to answer so their profile stands out and invites conversation.\n\n"
                    "Hinge prompts to choose from include:\n"
                    f"{HINGE_PROMPT_MENU}\n\n"
                    "Pick prompts that play to the member's strengths and give them room for a "
                    "concrete, story-driven answer.\n"
                    "</system>\n\n"
                    "<user>\n"
                    "I'd describe my vibe as {{vibe}}.\n\n"
                    "Recommend 3 Hinge prompts I should answer to stand out, and one line on why each "
                    "fits me.\n"
                    "</user>"
                ),
            },
        ],
    },
    {
        "name": "convo_reviver",
        "description": "Restart a Hinge match conversation that's gone quiet, without being needy.",
        "alias": {"production": 1},
        "versions": [
            {
                "description": "Initial version.",
                "template": (
                    "<system>\n"
                    "You help Hinge members restart a conversation that's gone quiet — in a light, "
                    "low-pressure way. Never needy, guilt-trippy, or \"did you ghost me?\". Suggest one "
                    "message that re-opens the chat and gives them an easy way to reply.\n"
                    "</system>\n\n"
                    "<user>\n"
                    "Our Hinge chat went quiet. Their last message was:\n"
                    "\"{{last_message}}\"\n\n"
                    "Write a {{tone}} message that revives the conversation.\n"
                    "</user>"
                ),
            },
        ],
    },
    {
        "name": "first_date_idea",
        "description": "Suggest first-date ideas for a Hinge match, tuned to vibe and city.",
        "alias": {"production": 1},
        "versions": [
            {
                "description": "Initial version.",
                "template": (
                    "<system>\n"
                    "You suggest first-date ideas for Hinge matches. Favor low-pressure spots where "
                    "conversation flows easily — avoid loud bars and movie theaters where you can't talk.\n"
                    "</system>\n\n"
                    "<user>\n"
                    "Suggest 2 first-date ideas in {{city}} for a {{vibe}} vibe. For each, add one line "
                    "on why it makes conversation easy.\n"
                    "</user>"
                ),
            },
        ],
    },
]


def cleanup():
    print("Removing existing prompts in schema...")
    for short in ALL_NAMES:
        name = fq(short)
        # drop known aliases first (best effort)
        for alias in ("production", "candidate"):
            try:
                client.delete_prompt_alias(name=name, alias=alias)
            except Exception:
                pass
        # drop versions
        try:
            vers = list(client.search_prompt_versions(name=name).prompt_versions)
            for v in vers:
                try:
                    client.delete_prompt_version(name=name, version=str(v.version))
                except Exception:
                    pass
        except Exception:
            pass
        # drop the prompt entity
        try:
            client.delete_prompt(name=name)
            print(f"  deleted {short}")
        except Exception:
            pass  # didn't exist — fine


def ensure_experiment() -> str | None:
    try:
        exp = mlflow.get_experiment_by_name(EXPERIMENT)
        if exp:
            return exp.experiment_id
        return mlflow.create_experiment(EXPERIMENT)
    except Exception as e:
        print(f"  WARNING: experiment: {e}")
        return None


def tag_experiment(name: str, exp_id: str) -> None:
    try:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient(profile=PROFILE)
        w.api_client.do(
            "POST",
            f"/api/2.0/mlflow/unity-catalog/prompts/{name}/tags",
            body={"key": "_mlflow_experiment_ids", "value": f",{exp_id},"},
        )
    except Exception:
        pass


def seed():
    print(f"Seeding into {CATALOG}.{SCHEMA} via profile '{PROFILE}'\n")
    cleanup()
    exp_id = ensure_experiment()
    print(f"\nExperiment: {EXPERIMENT} (id={exp_id})")

    for spec in PROMPTS:
        name = fq(spec["name"])
        print(f"\n• {spec['name']}")
        try:
            client.create_prompt(name=name, description=spec["description"])
        except Exception as e:
            print(f"  ERROR creating prompt: {e}")
            continue
        for ver in spec["versions"]:
            try:
                pv = client.create_prompt_version(
                    name=name, template=ver["template"], description=ver["description"]
                )
                print(f"  v{pv.version} created")
            except Exception as e:
                print(f"  version skipped: {e}")
        for alias, ver_idx in spec.get("alias", {}).items():
            try:
                client.set_prompt_alias(name=name, alias=alias, version=ver_idx)
                print(f"  alias @{alias} -> v{ver_idx}")
            except Exception as e:
                print(f"  alias @{alias} skipped: {e}")
        if exp_id:
            tag_experiment(name, exp_id)

    print("\nDone.")


if __name__ == "__main__":
    seed()
