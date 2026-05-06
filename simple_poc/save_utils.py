# simple_poc/save_utils.py
"""
Persistence utilities for Kaggle sessions.
Handles push to HuggingFace Hub and local zip packaging.
"""
import os
import shutil
import glob
import json
from pathlib import Path


def push_to_hub(local_path: str, repo_id: str, path_in_repo: str,
                token: str = None, commit_message: str = "update"):
    """
    Push a single file to HuggingFace Hub dataset repo.
    Silently skips if huggingface_hub is not authenticated.
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj = local_path,
            path_in_repo    = path_in_repo,
            repo_id         = repo_id,
            repo_type       = "dataset",
            token           = token,
            commit_message  = commit_message,
        )
        print(f"  ☁️  Pushed → {repo_id}/{path_in_repo}")
    except Exception as e:
        print(f"  ⚠️  HF push failed for {local_path}: {e}")


def push_directory_to_hub(local_dir: str, repo_id: str,
                           prefix_in_repo: str = "",
                           token: str = None,
                           extensions: tuple = ('.json', '.pth')):
    """
    Push all files matching extensions from local_dir to HF Hub,
    preserving relative directory structure under prefix_in_repo.
    """
    from huggingface_hub import HfApi
    api   = HfApi()
    files = [f for ext in extensions
             for f in glob.glob(os.path.join(local_dir, '**', f'*{ext}'),
                                recursive=True)]

    if not files:
        print(f"  ⚠️  No files found in {local_dir}")
        return

    print(f"  ☁️  Pushing {len(files)} files to {repo_id}/{prefix_in_repo}...")
    for f in files:
        rel = os.path.relpath(f, local_dir)
        path_in_repo = os.path.join(prefix_in_repo, rel) if prefix_in_repo else rel
        try:
            api.upload_file(
                path_or_fileobj = f,
                path_in_repo    = path_in_repo,
                repo_id         = repo_id,
                repo_type       = "dataset",
                token           = token,
                commit_message  = f"batch update: {rel}",
            )
        except Exception as e:
            print(f"    ⚠️  Failed {rel}: {e}")
    print(f"  ✅ Push complete.")


def pull_from_hub(repo_id: str, path_in_repo: str,
                  local_path: str, token: str = None):
    """
    Download a single file from HF Hub to local_path.
    """
    try:
        from huggingface_hub import hf_hub_download
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        downloaded = hf_hub_download(
            repo_id   = repo_id,
            filename  = path_in_repo,
            repo_type = "dataset",
            token     = token,
            local_dir = os.path.dirname(local_path),
        )
        shutil.copy(downloaded, local_path)
        print(f"  ⬇️  Pulled {path_in_repo} → {local_path}")
    except Exception as e:
        print(f"  ⚠️  HF pull failed for {path_in_repo}: {e}")


def pull_directory_from_hub(repo_id: str, prefix_in_repo: str,
                             local_dir: str, token: str = None):
    """
    Download all files under prefix_in_repo from HF Hub to local_dir.
    Use at session start to restore previous progress.
    """
    try:
        from huggingface_hub import snapshot_download
        print(f"  ⬇️  Pulling {repo_id}/{prefix_in_repo} → {local_dir}...")
        snapshot_download(
            repo_id        = repo_id,
            repo_type      = "dataset",
            allow_patterns = [f"{prefix_in_repo}/**"],
            local_dir      = local_dir,
            token          = token,
        )
        print(f"  ✅ Restore complete.")
    except Exception as e:
        print(f"  ⚠️  HF pull failed: {e}")


def zip_and_package(source_dir: str, output_name: str,
                    output_base: str = "/kaggle/working") -> str:
    """
    Zip source_dir and save to output_base.
    Returns path to the zip file.
    """
    zip_path = os.path.join(output_base, output_name)
    if os.path.exists(f"{zip_path}.zip"):
        os.remove(f"{zip_path}.zip")
    shutil.make_archive(zip_path, 'zip', source_dir)
    size_mb = os.path.getsize(f"{zip_path}.zip") / 1e6
    print(f"  📦 Zipped {source_dir} → {zip_path}.zip ({size_mb:.1f} MB)")
    return f"{zip_path}.zip"