#!/usr/bin/env python3
"""
bootstrap_spec_vectorstore.py

Minimal “first mile”:
- Pull spec.json from GitHub
- Upload to an OpenAI Vector Store
- Attach to a GPT Assistant with File Search enabled

Env vars (required *)
  * GITHUB_TOKEN        Fine-grained PAT with repo read on the target repo
  * OPENAI_API_KEY      OpenAI API key

  GITHUB_OWNER          GitHub owner/org (default: from --owner)
  GITHUB_REPO           Repo name (default: from --repo)
  GITHUB_PATH           Path to spec (default: spec/spec.json)
  GITHUB_REF            Branch/SHA (default: main)

  VECTOR_STORE_ID       If set, reuse this vector store; else create a new one
  VECTOR_STORE_NAME     Name when creating (default: "Boat Spec (MVP)")
  ASSISTANT_ID          If set, reuse; else create a new assistant
  ASSISTANT_NAME        Name when creating (default: "Boat Spec Assistant (MVP)")
  MODEL                 Model for assistant (default: gpt-4o-mini)

Usage (local):
  export GITHUB_TOKEN=ghp_...
  export OPENAI_API_KEY=sk-...
  python bootstrap_spec_vectorstore.py --owner youruser --repo boat-project

In Render:
  Add the env vars on your service → Environment → add variables → deploy.
"""

import argparse
import os
import sys
import requests
from typing import Optional

# OpenAI SDK v1.x
try:
    from openai import OpenAI
except Exception as e:
    print("OpenAI SDK is required. pip install openai", file=sys.stderr)
    raise

import openai as openai_pkg, sys
print("OpenAI SDK version at runtime:", getattr(openai_pkg, "__version__", "unknown"), " | Python:", sys.version)

RAW_ACCEPT = "application/vnd.github.raw"
JSON_ACCEPT = "application/vnd.github+json"
GH_API_VER = "2022-11-28"


def fetch_spec_from_github(owner: str, repo: str, path: str, ref: str, token: str) -> bytes:
    """
    Use the GitHub Contents API with Accept: application/vnd.github.raw
    so we get raw bytes directly (no base64 decoding).
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": RAW_ACCEPT,
        "X-GitHub-Api-Version": GH_API_VER,
        "User-Agent": "spec-vectorstore-bootstrap",
    }
    params = {"ref": ref} if ref else {}
    r = requests.get(url, headers=headers, params=params, timeout=30)
    if r.status_code == 404:
        raise FileNotFoundError(f"GitHub path not found: {owner}/{repo}:{path}@{ref}")
    r.raise_for_status()
    return r.content


def ensure_vector_store(client: OpenAI, vector_store_id: Optional[str], name: str) -> str:
    if vector_store_id:
        print(f"Reusing Vector Store: {vector_store_id}")
        return vector_store_id
    vs = client.beta.vector_stores.create(name=name)
    print(f"Created Vector Store: {vs.id} (name='{name}')")
    return vs.id


def upload_spec_to_vector_store(client: OpenAI, vector_store_id: str, spec_bytes: bytes, filename: str = "spec.json"):
    """
    Upload as a stream. The OpenAI API infers file type from the filename extension,
    so keep filename='spec.json'.
    """
    import io
    bio = io.BytesIO(spec_bytes)
    bio.name = filename  # important: gives the stream an extension for type detection
    batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store_id,
        files=[bio],
    )
    if batch.status != "completed":
        raise RuntimeError(f"Vector store file batch not completed: status={batch.status}, counts={batch.file_counts}")
    print(f"Uploaded to Vector Store. File counts: {batch.file_counts}")


def ensure_assistant_with_fs(client: OpenAI, assistant_id: Optional[str], vs_id: str, name: str, model: str) -> str:
    if assistant_id:
        # Attach the vector store to an existing assistant by updating tool_resources
        a = client.beta.assistants.update(
            assistant_id,
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [vs_id]}},
        )
        print(f"Updated Assistant: {a.id} (attached Vector Store {vs_id})")
        return a.id
    # Create new assistant
    a = client.beta.assistants.create(
        model=model,
        name=name,
        instructions=(
            "Answer questions about the boat using File Search. "
            "Cite facts from the provided spec where possible."
        ),
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vs_id]}},
    )
    print(f"Created Assistant: {a.id} (model={model}, attached Vector Store {vs_id})")
    return a.id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True, help="GitHub owner/org")
    ap.add_argument("--repo", required=True, help="GitHub repo")
    ap.add_argument("--path", default=os.environ.get("GITHUB_PATH", "spec/spec.json"))
    ap.add_argument("--ref", default=os.environ.get("GITHUB_REF", "main"))
    args = ap.parse_args()

    gh_token = os.environ.get("GITHUB_TOKEN")
    if not gh_token:
        print("Missing env GITHUB_TOKEN", file=sys.stderr)
        sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing env OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    owner = os.environ.get("GITHUB_OWNER", args.owner)
    repo = os.environ.get("GITHUB_REPO", args.repo)
    path = os.environ.get("GITHUB_PATH", args.path)
    ref = os.environ.get("GITHUB_REF", args.ref)

    # 1) Fetch spec bytes
    print(f"Fetching {owner}/{repo}:{path}@{ref} from GitHub…")
    spec_bytes = fetch_spec_from_github(owner, repo, path, ref, gh_token)
    print(f"Fetched {len(spec_bytes)} bytes.")

    # 2) Create/reuse Vector Store and upload
    client = OpenAI()
    vs_id = ensure_vector_store(
        client,
        os.environ.get("VECTOR_STORE_ID"),
        os.environ.get("VECTOR_STORE_NAME", "Boat Spec (MVP)"),
    )
    upload_spec_to_vector_store(client, vs_id, spec_bytes, filename=os.path.basename(path) or "spec.json")

    # 3) Create/update Assistant wired to that store
    assistant_id = ensure_assistant_with_fs(
        client,
        os.environ.get("ASSISTANT_ID"),
        vs_id,
        os.environ.get("ASSISTANT_NAME", "Boat Spec Assistant (MVP)"),
        os.environ.get("MODEL", "gpt-4o-mini"),
    )

    # 4) Print ready-to-use info
    print("\n✅ Ready!")
    print(f"Vector Store ID : {vs_id}")
    print(f"Assistant ID    : {assistant_id}")
    print("You can now ask the assistant questions grounded in spec.json.")


if __name__ == "__main__":
    main()
