"""Centralized cache management with manifest index for human-readable metadata."""

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config import EMBEDDING_CACHE_MAX_AGE_DAYS

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")
EMBEDDING_CACHE_DIR = Path("data/embedding_cache")
CLASSIFICATION_CACHE_DIR = Path("data/classification_cache")
MANIFEST_PATH = CACHE_DIR / "manifest.json"


def _cache_key(text: str, model: str, prompt_version: str) -> str:
    """Return SHA-256 hash of text + model + prompt version for cache keying."""
    return hashlib.sha256((text + model + prompt_version).encode()).hexdigest()


def _embedding_cache_key(text: str, model: str) -> str:
    """Return SHA-256 hash of text + model for embedding cache keying."""
    return hashlib.sha256((text + model).encode()).hexdigest()


def _classification_cache_key(text: str, model: str, prompt_version: str) -> str:
    """Return SHA-256 hash of text + model + prompt_version for classification cache keying."""
    return hashlib.sha256(
        (text + "||" + model + "||" + prompt_version).encode()
    ).hexdigest()


class CacheManager:
    """Manages extraction and embedding caches with a manifest index."""

    def __init__(self):
        self._manifest: dict | None = None

    # ── Manifest CRUD ──────────────────────────────────────────────

    def _load_manifest(self) -> dict:
        if self._manifest is not None:
            return self._manifest
        if MANIFEST_PATH.exists():
            self._manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        else:
            self._manifest = {"version": 1, "entries": {}}
            self.migrate_legacy()
        return self._manifest

    def _save_manifest(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(
            json.dumps(self._manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── Extraction cache ───────────────────────────────────────────

    def get_extraction(self, text: str, model: str, prompt_version: str) -> dict | None:
        """Load cached extraction result, or None if not cached."""
        key = _cache_key(text, model, prompt_version)
        cache_file = CACHE_DIR / f"{key}.json"
        if cache_file.exists():
            # Lazy-add to manifest if missing
            manifest = self._load_manifest()
            if key not in manifest["entries"]:
                self._index_extraction_file(cache_file, key)
            return json.loads(cache_file.read_text(encoding="utf-8"))
        return None

    def put_extraction(
        self,
        text: str,
        model: str,
        prompt_version: str,
        result: dict,
        document_name: str | None = None,
    ) -> None:
        """Save extraction result and update manifest."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key = _cache_key(text, model, prompt_version)
        cache_file = CACHE_DIR / f"{key}.json"
        cache_file.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Count chunks for this hash
        chunk_count = len(list(CACHE_DIR.glob(f"{key}_chunk*.json")))
        konsep_count = len(result.get("konsep", []))

        manifest = self._load_manifest()
        manifest["entries"][key] = {
            "document_name": document_name,
            "model": model,
            "prompt_version": prompt_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "chunk_count": chunk_count,
            "konsep_count": konsep_count,
            "partial": bool(result.get("_partial")),
        }
        self._save_manifest()
        logger.info("Cached extraction result %s", key[:12])

    def get_chunk(self, base_hash: str, chunk_index: int) -> dict | None:
        """Load a cached chunk, or None."""
        chunk_file = CACHE_DIR / f"{base_hash}_chunk{chunk_index}.json"
        if chunk_file.exists():
            return json.loads(chunk_file.read_text(encoding="utf-8"))
        return None

    def put_chunk(self, base_hash: str, chunk_index: int, data: dict) -> None:
        """Save a chunk to cache."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        chunk_file = CACHE_DIR / f"{base_hash}_chunk{chunk_index}.json"
        chunk_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def get_extraction_hash(self, text: str, model: str, prompt_version: str) -> str:
        """Return the cache key hash for given parameters."""
        return _cache_key(text, model, prompt_version)

    # ── Embedding cache ────────────────────────────────────────────

    def get_embedding(self, text: str, model: str) -> list[float] | None:
        """Load cached embedding, or None."""
        key = _embedding_cache_key(text, model)
        cache_file = EMBEDDING_CACHE_DIR / f"{key}.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return data.get("embedding")
        return None

    def put_embedding(self, text: str, model: str, embedding: list[float]) -> None:
        """Save embedding to cache."""
        EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key = _embedding_cache_key(text, model)
        cache_file = EMBEDDING_CACHE_DIR / f"{key}.json"
        cache_file.write_text(
            json.dumps({"embedding": embedding}, indent=2), encoding="utf-8"
        )

    # ── Classification cache ───────────────────────────────────────

    def get_classification(
        self, text: str, model: str, prompt_version: str
    ) -> dict | None:
        """Load cached classification result, or None."""
        key = _classification_cache_key(text, model, prompt_version)
        cache_file = CLASSIFICATION_CACHE_DIR / f"{key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))
        return None

    def put_classification(
        self, text: str, model: str, prompt_version: str, result: dict
    ) -> None:
        """Save classification result to cache."""
        CLASSIFICATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key = _classification_cache_key(text, model, prompt_version)
        cache_file = CLASSIFICATION_CACHE_DIR / f"{key}.json"
        cache_file.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── Management ─────────────────────────────────────────────────

    def list_extractions(self) -> list[dict]:
        """Return human-readable list of cached extractions."""
        manifest = self._load_manifest()
        entries = []
        for hash_key, meta in manifest["entries"].items():
            doc = meta.get("document_name") or "Unknown"
            model = meta.get("model", "unknown")
            # Short model name (after /)
            model_short = model.split("/")[-1] if "/" in model else model
            pv = meta.get("prompt_version", "?")
            created = meta.get("created_at", "")
            date_str = created[:10] if created else "?"
            label = f"{doc} ({model_short}, {pv}, {date_str})"
            entries.append(
                {
                    "hash": hash_key,
                    "label": label,
                    "document_name": doc,
                    "model": model,
                    "prompt_version": pv,
                    "created_at": created,
                    "chunk_count": meta.get("chunk_count", 0),
                    "konsep_count": meta.get("konsep_count", 0),
                    "partial": meta.get("partial", False),
                }
            )
        return entries

    def delete_extraction(self, hash_key: str) -> int:
        """Delete extraction and all its chunk files. Returns count of files deleted."""
        deleted = 0
        # Delete final result
        final = CACHE_DIR / f"{hash_key}.json"
        if final.exists():
            final.unlink()
            deleted += 1
        # Delete chunk files
        for chunk_file in CACHE_DIR.glob(f"{hash_key}_chunk*.json"):
            chunk_file.unlink()
            deleted += 1
        # Remove from manifest
        manifest = self._load_manifest()
        manifest["entries"].pop(hash_key, None)
        self._save_manifest()
        logger.info("Deleted extraction %s (%d files)", hash_key[:12], deleted)
        return deleted

    def get_stats(self) -> dict:
        """Return cache statistics."""
        manifest = self._load_manifest()
        extraction_count = len(manifest["entries"])

        # Extraction cache size
        extraction_size = 0
        extraction_files = 0
        if CACHE_DIR.exists():
            for f in CACHE_DIR.iterdir():
                if f.suffix == ".json" and f.name != "manifest.json":
                    extraction_size += f.stat().st_size
                    extraction_files += 1

        # Embedding cache size
        embedding_size = 0
        embedding_files = 0
        if EMBEDDING_CACHE_DIR.exists():
            for f in EMBEDDING_CACHE_DIR.iterdir():
                if f.suffix == ".json":
                    embedding_size += f.stat().st_size
                    embedding_files += 1

        return {
            "extraction_count": extraction_count,
            "extraction_files": extraction_files,
            "extraction_size_mb": round(extraction_size / (1024 * 1024), 2),
            "embedding_files": embedding_files,
            "embedding_size_mb": round(embedding_size / (1024 * 1024), 2),
            "total_size_mb": round(
                (extraction_size + embedding_size) / (1024 * 1024), 2
            ),
        }

    def cleanup_embeddings(self, max_age_days: int | None = None) -> int:
        """Delete embedding cache files older than max_age_days. Returns count deleted."""
        max_age_days = max_age_days or EMBEDDING_CACHE_MAX_AGE_DAYS
        if not EMBEDDING_CACHE_DIR.exists():
            return 0
        cutoff = time.time() - (max_age_days * 86400)
        deleted = 0
        for f in EMBEDDING_CACHE_DIR.iterdir():
            if f.suffix == ".json" and f.stat().st_mtime < cutoff:
                f.unlink()
                deleted += 1
        logger.info("Cleaned up %d old embedding cache files", deleted)
        return deleted

    def clear_all(self) -> dict:
        """Delete all extraction and embedding cache files. Returns counts deleted."""
        extraction_deleted = 0
        if CACHE_DIR.exists():
            for f in CACHE_DIR.iterdir():
                if f.suffix == ".json":
                    f.unlink()
                    extraction_deleted += 1
        # Reset manifest
        self._manifest = None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        embedding_deleted = 0
        if EMBEDDING_CACHE_DIR.exists():
            for f in EMBEDDING_CACHE_DIR.iterdir():
                if f.suffix == ".json":
                    f.unlink()
                    embedding_deleted += 1

        classification_deleted = 0
        if CLASSIFICATION_CACHE_DIR.exists():
            for f in CLASSIFICATION_CACHE_DIR.iterdir():
                if f.suffix == ".json":
                    f.unlink()
                    classification_deleted += 1

        logger.info(
            "Cleared all caches: %d extraction files, %d embedding files, %d classification files",
            extraction_deleted,
            embedding_deleted,
            classification_deleted,
        )
        return {
            "extraction_deleted": extraction_deleted,
            "embedding_deleted": embedding_deleted,
            "classification_deleted": classification_deleted,
        }

    def migrate_legacy(self) -> None:
        """Index existing cache files that aren't in the manifest."""
        if not CACHE_DIR.exists():
            return
        manifest = self._load_manifest()
        count = 0
        for f in sorted(CACHE_DIR.glob("*.json")):
            if f.name == "manifest.json" or "_chunk" in f.name:
                continue
            hash_key = f.stem
            if hash_key not in manifest["entries"]:
                self._index_extraction_file(f, hash_key)
                count += 1
        if count:
            self._save_manifest()
            logger.info("Migrated %d legacy cache entries to manifest", count)

    def _index_extraction_file(self, path: Path, hash_key: str) -> None:
        """Add an existing cache file to the manifest with best-effort metadata."""
        manifest = self._load_manifest()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        chunk_count = len(list(CACHE_DIR.glob(f"{hash_key}_chunk*.json")))
        konsep_count = len(data.get("konsep", data.get("topics", [])))
        mtime = datetime.fromtimestamp(
            os.path.getmtime(path), tz=timezone.utc
        ).isoformat()

        manifest["entries"][hash_key] = {
            "document_name": None,
            "model": "unknown",
            "prompt_version": "unknown",
            "created_at": mtime,
            "chunk_count": chunk_count,
            "konsep_count": konsep_count,
            "partial": bool(data.get("_partial")),
        }


# Module-level singleton
cache_manager = CacheManager()
