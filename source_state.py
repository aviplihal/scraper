"""Persistent source-state management for approved and discovered lead sources."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

SOURCE_ACCURACY_PRESETS: dict[str, dict[str, int]] = {
    "strict": {
        "approved_threshold": 92,
        "temporary_seed_threshold": 86,
        "review_threshold": 80,
        "approved_gap": -3,
        "temporary_seed_gap": -6,
        "review_gap": -10,
    },
    "balanced": {
        "approved_threshold": 90,
        "temporary_seed_threshold": 80,
        "review_threshold": 68,
        "approved_gap": -5,
        "temporary_seed_gap": -10,
        "review_gap": -15,
    },
    "aggressive": {
        "approved_threshold": 87,
        "temporary_seed_threshold": 74,
        "review_threshold": 66,
        "approved_gap": -8,
        "temporary_seed_gap": -16,
        "review_gap": -22,
    },
}

_SOCIAL_DOMAINS = {
    "linkedin": "linkedin.com",
    "x": "x.com",
}


def normalize_domain(value: str) -> str:
    """Normalize a website/domain string to a bare lowercase host."""
    parsed = urlparse(value if "://" in value else f"https://{value}")
    host = parsed.netloc or parsed.path
    return host.lower().split(":", 1)[0].removeprefix("www.")


def normalize_platform(value: str) -> str:
    """Normalize a social-platform name."""
    return str(value).strip().lower()


def source_key(kind: str, identifier: str) -> str:
    """Build the persistent metadata key for a source."""
    return f"{kind}:{identifier}"


def domain_for_platform(platform: str) -> str:
    """Return the canonical domain for a supported social platform."""
    return _SOCIAL_DOMAINS.get(normalize_platform(platform), normalize_platform(platform))


def platform_for_domain(domain: str) -> str | None:
    """Map a domain to its supported social-platform name, if any."""
    normalized = normalize_domain(domain)
    for platform, social_domain in _SOCIAL_DOMAINS.items():
        if normalized == social_domain:
            return platform
    return None


def seed_approved_sources_from_config(config: dict[str, Any]) -> dict[str, list[str]]:
    """Seed the approved-source pool from config with backward compatibility."""
    approved = config.get("approved_sources")
    if isinstance(approved, dict):
        return {
            "web_domains": [
                normalize_domain(value)
                for value in approved.get("web_domains", [])
                if str(value).strip()
            ],
            "social_platforms": [
                normalize_platform(value)
                for value in approved.get("social_platforms", [])
                if str(value).strip()
            ],
        }

    seeded = {"web_domains": [], "social_platforms": []}
    website = str(config.get("website", "NA")).strip()
    if website and website.upper() != "NA":
        seeded["web_domains"].append(normalize_domain(website))
    for platform in config.get("social_platforms", []):
        if str(platform).strip():
            seeded["social_platforms"].append(normalize_platform(platform))
    return seeded


def infer_source_identity(url: str) -> tuple[str, str]:
    """Infer source kind and identifier from a URL."""
    domain = normalize_domain(url)
    platform = platform_for_domain(domain)
    if platform:
        return "social_platform", platform
    return "web_domain", domain


def infer_source_family(kind: str, identifier: str) -> str:
    """Infer a discovery family hint for a source."""
    normalized = identifier.lower()
    if kind == "social_platform":
        return "people_search"
    if normalized in {"github.com", "gitlab.com"}:
        return "developer_profiles"
    if normalized in {"ycombinator.com"}:
        return "people_directory"
    return "generic_web"


class SourceState:
    """Persist approved, temporary, rejected, and pending-review source pools."""

    def __init__(self, client_id: str, config: dict[str, Any]):
        self.client_id = client_id
        self.config = config
        self.path = Path(f"state/{client_id}_source_state.json")
        self.review_path = Path(f"state/{client_id}_source_review.json")
        self._data: dict[str, Any] = {}
        self._load()
        self._seed_from_config(config)
        self._save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _default(self) -> dict[str, Any]:
        return {
            "approved_sources": {"web_domains": [], "social_platforms": []},
            "temporary_seed_sources": {"web_domains": [], "social_platforms": []},
            "rejected_sources": {"web_domains": [], "social_platforms": []},
            "pending_review_sources": {"web_domains": [], "social_platforms": []},
            "source_metadata": {},
        }

    def _load(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with open(self.path) as f:
                self._data = json.load(f)
        else:
            self._data = self._default()
        self._normalize_loaded_state()

    def _normalize_loaded_state(self) -> None:
        default = self._default()
        for key in (
            "approved_sources",
            "temporary_seed_sources",
            "rejected_sources",
            "pending_review_sources",
        ):
            raw_group = self._data.get(key, {})
            default_group = default[key]
            group = {
                "web_domains": [
                    normalize_domain(value)
                    for value in raw_group.get("web_domains", default_group["web_domains"])
                    if str(value).strip()
                ],
                "social_platforms": [
                    normalize_platform(value)
                    for value in raw_group.get("social_platforms", default_group["social_platforms"])
                    if str(value).strip()
                ],
            }
            group["web_domains"] = list(dict.fromkeys(group["web_domains"]))
            group["social_platforms"] = list(dict.fromkeys(group["social_platforms"]))
            self._data[key] = group
        self._data.setdefault("source_metadata", {})

    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2, sort_keys=True)

    def _seed_from_config(self, config: dict[str, Any]) -> None:
        seeded = seed_approved_sources_from_config(config)
        for domain in seeded["web_domains"]:
            self._ensure_in_pool("approved_sources", "web_domains", domain)
            self._update_metadata("web_domain", domain, family=infer_source_family("web_domain", domain))
        for platform in seeded["social_platforms"]:
            self._ensure_in_pool("approved_sources", "social_platforms", platform)
            self._update_metadata(
                "social_platform",
                platform,
                family=infer_source_family("social_platform", platform),
            )

    # ------------------------------------------------------------------
    # Pool access
    # ------------------------------------------------------------------

    def approved_sources(self) -> dict[str, list[str]]:
        return json.loads(json.dumps(self._data["approved_sources"]))

    def temporary_seed_sources(self) -> dict[str, list[str]]:
        return json.loads(json.dumps(self._data["temporary_seed_sources"]))

    def rejected_sources(self) -> dict[str, list[str]]:
        return json.loads(json.dumps(self._data["rejected_sources"]))

    def pending_review_sources(self) -> dict[str, list[str]]:
        return json.loads(json.dumps(self._data["pending_review_sources"]))

    def active_pass1_sources(self) -> dict[str, list[str]]:
        return {
            "web_domains": list(
                dict.fromkeys(
                    self._data["approved_sources"]["web_domains"]
                    + self._data["temporary_seed_sources"]["web_domains"]
                )
            ),
            "social_platforms": list(
                dict.fromkeys(
                    self._data["approved_sources"]["social_platforms"]
                    + self._data["temporary_seed_sources"]["social_platforms"]
                )
            ),
        }

    def has_pass1_sources_for_mode(self, source_mode: str) -> bool:
        groups = self.active_pass1_sources()
        if source_mode == "web":
            return bool(groups["web_domains"])
        if source_mode == "human_emulator":
            return bool(groups["social_platforms"])
        return bool(groups["web_domains"] or groups["social_platforms"])

    def source_status(self, kind: str, identifier: str) -> str:
        bucket_key = "web_domains" if kind == "web_domain" else "social_platforms"
        normalized = normalize_domain(identifier) if kind == "web_domain" else normalize_platform(identifier)
        for pool_name, status in (
            ("approved_sources", "approved"),
            ("temporary_seed_sources", "temporary_seed"),
            ("pending_review_sources", "pending_review"),
            ("rejected_sources", "rejected"),
        ):
            if normalized in self._data[pool_name][bucket_key]:
                return status
        return "discovered"

    def family_hints(self, include_temporary: bool = True) -> set[str]:
        families: set[str] = set()
        pools = ["approved_sources"]
        if include_temporary:
            pools.append("temporary_seed_sources")
        for pool_name in pools:
            for domain in self._data[pool_name]["web_domains"]:
                families.add(self.metadata_for("web_domain", domain).get("family", infer_source_family("web_domain", domain)))
            for platform in self._data[pool_name]["social_platforms"]:
                families.add(
                    self.metadata_for("social_platform", platform).get(
                        "family",
                        infer_source_family("social_platform", platform),
                    )
                )
        return {family for family in families if family}

    def metadata_for(self, kind: str, identifier: str) -> dict[str, Any]:
        key = source_key(kind, identifier)
        return dict(self._data["source_metadata"].get(key, {}))

    def is_temp_seed(self, kind: str, identifier: str) -> bool:
        return self.source_status(kind, identifier) == "temporary_seed"

    # ------------------------------------------------------------------
    # Promotion / rejection / review
    # ------------------------------------------------------------------

    def promote_approved(self, kind: str, identifier: str, family: str, score: int) -> None:
        bucket_key = "web_domains" if kind == "web_domain" else "social_platforms"
        normalized = normalize_domain(identifier) if kind == "web_domain" else normalize_platform(identifier)
        self._remove_from_all_nonapproved_pools(bucket_key, normalized)
        self._ensure_in_pool("approved_sources", bucket_key, normalized)
        self._update_metadata(
            kind,
            normalized,
            family=family,
            last_score=score,
            last_outcome="approved",
            exhausted=False,
            dry_runs_without_new_leads=0,
        )
        self._save()

    def promote_temporary_seed(self, kind: str, identifier: str, family: str, score: int) -> None:
        bucket_key = "web_domains" if kind == "web_domain" else "social_platforms"
        normalized = normalize_domain(identifier) if kind == "web_domain" else normalize_platform(identifier)
        self._remove_from_pool("pending_review_sources", bucket_key, normalized)
        self._remove_from_pool("rejected_sources", bucket_key, normalized)
        self._ensure_in_pool("temporary_seed_sources", bucket_key, normalized)
        self._update_metadata(
            kind,
            normalized,
            family=family,
            last_score=score,
            last_outcome="temporary_seed",
            exhausted=False,
        )
        self._save()

    def reject_source(self, kind: str, identifier: str, family: str, score: int) -> None:
        bucket_key = "web_domains" if kind == "web_domain" else "social_platforms"
        normalized = normalize_domain(identifier) if kind == "web_domain" else normalize_platform(identifier)
        self._remove_from_pool("pending_review_sources", bucket_key, normalized)
        self._remove_from_pool("temporary_seed_sources", bucket_key, normalized)
        self._ensure_in_pool("rejected_sources", bucket_key, normalized)
        self._update_metadata(
            kind,
            normalized,
            family=family,
            last_score=score,
            last_outcome="rejected",
        )
        self._save()

    def queue_for_review(
        self,
        kind: str,
        identifier: str,
        family: str,
        score: int,
        candidate_leads: list[dict[str, Any]],
        baseline_leads: list[dict[str, Any]],
    ) -> None:
        bucket_key = "web_domains" if kind == "web_domain" else "social_platforms"
        normalized = normalize_domain(identifier) if kind == "web_domain" else normalize_platform(identifier)
        self._remove_from_pool("rejected_sources", bucket_key, normalized)
        self._ensure_in_pool("pending_review_sources", bucket_key, normalized)
        self._update_metadata(
            kind,
            normalized,
            family=family,
            last_score=score,
            last_outcome="pending_review",
        )
        self._append_review_entry(
            {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "source_kind": kind,
                "source_identifier": normalized,
                "family": family,
                "score": score,
                "outcome": "pending_review",
                "candidate_leads": candidate_leads[:3],
                "baseline_leads": baseline_leads[:3],
            }
        )
        self._save()

    def mark_temporary_seed_exhausted(self, kind: str, identifier: str, reason: str = "") -> None:
        bucket_key = "web_domains" if kind == "web_domain" else "social_platforms"
        normalized = normalize_domain(identifier) if kind == "web_domain" else normalize_platform(identifier)
        self._remove_from_pool("temporary_seed_sources", bucket_key, normalized)
        self._update_metadata(
            kind,
            normalized,
            exhausted=True,
            last_outcome="temporary_seed_exhausted",
            exhaustion_reason=reason,
        )
        self._save()

    # ------------------------------------------------------------------
    # Run finalization
    # ------------------------------------------------------------------

    def finalize_run(self, run_stats: dict[str, dict[str, Any]]) -> None:
        """Update per-source usage stats and retire exhausted temporary seeds."""
        changed = False
        for meta_key, stats in run_stats.items():
            if meta_key not in self._data["source_metadata"]:
                continue
            metadata = self._data["source_metadata"][meta_key]
            metadata["last_used_at"] = datetime.now(timezone.utc).isoformat()
            metadata["last_run_saved_count"] = int(stats.get("saved_count", 0))
            metadata["last_run_duplicate_count"] = int(stats.get("duplicate_count", 0))
            metadata["last_run_rejected_count"] = int(stats.get("rejected_count", 0))
            metadata["last_run_fetch_count"] = int(stats.get("fetch_count", 0))
            metadata["total_saved_count"] = int(metadata.get("total_saved_count", 0)) + int(stats.get("saved_count", 0))
            metadata["total_duplicate_count"] = int(metadata.get("total_duplicate_count", 0)) + int(stats.get("duplicate_count", 0))
            metadata["total_rejected_count"] = int(metadata.get("total_rejected_count", 0)) + int(stats.get("rejected_count", 0))
            changed = True

            source_kind, identifier = meta_key.split(":", 1)
            if self.is_temp_seed(source_kind, identifier):
                saved_count = int(stats.get("saved_count", 0))
                fetch_count = int(stats.get("fetch_count", 0))
                duplicate_count = int(stats.get("duplicate_count", 0))
                rejected_count = int(stats.get("rejected_count", 0))
                if saved_count > 0:
                    metadata["dry_runs_without_new_leads"] = 0
                    metadata["exhausted"] = False
                elif fetch_count > 0 and (duplicate_count > 0 or rejected_count > 0):
                    dry_runs = int(metadata.get("dry_runs_without_new_leads", 0)) + 1
                    metadata["dry_runs_without_new_leads"] = dry_runs
                    if dry_runs >= 2:
                        self.mark_temporary_seed_exhausted(
                            source_kind,
                            identifier,
                            reason="Repeated runs produced no new viable leads.",
                        )
                else:
                    metadata.setdefault("dry_runs_without_new_leads", 0)

        if changed:
            self._save()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_in_pool(self, pool_name: str, bucket_key: str, value: str) -> None:
        bucket = self._data[pool_name][bucket_key]
        if value not in bucket:
            bucket.append(value)

    def _remove_from_pool(self, pool_name: str, bucket_key: str, value: str) -> None:
        bucket = self._data[pool_name][bucket_key]
        if value in bucket:
            bucket.remove(value)

    def _remove_from_all_nonapproved_pools(self, bucket_key: str, value: str) -> None:
        for pool_name in ("temporary_seed_sources", "rejected_sources", "pending_review_sources"):
            self._remove_from_pool(pool_name, bucket_key, value)

    def _update_metadata(self, kind: str, identifier: str, **updates: Any) -> None:
        key = source_key(kind, identifier)
        metadata = self._data["source_metadata"].setdefault(
            key,
            {
                "kind": kind,
                "identifier": identifier,
                "family": infer_source_family(kind, identifier),
                "last_score": None,
                "last_outcome": "",
                "dry_runs_without_new_leads": 0,
                "exhausted": False,
            },
        )
        metadata.update(updates)

    def _append_review_entry(self, entry: dict[str, Any]) -> None:
        self.review_path.parent.mkdir(parents=True, exist_ok=True)
        review_entries: list[dict[str, Any]]
        if self.review_path.exists():
            with open(self.review_path) as f:
                review_entries = json.load(f)
        else:
            review_entries = []

        source_kind = entry["source_kind"]
        source_identifier = entry["source_identifier"]
        review_entries = [
            existing
            for existing in review_entries
            if not (
                existing.get("source_kind") == source_kind
                and existing.get("source_identifier") == source_identifier
            )
        ]
        review_entries.append(entry)
        with open(self.review_path, "w") as f:
            json.dump(review_entries, f, indent=2, sort_keys=True)
