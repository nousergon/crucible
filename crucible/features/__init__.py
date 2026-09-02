"""The feature layer: one materialized, hashed artifact between data and slots.

Normative source: plan §10 component 4.

Without it, R and M recompute features from different code and "the signal
degraded" cannot be separated from "the feature changed". With it, both slots
record the SAME input hash in their manifests, and that identity is an
acceptance clause rather than a convention.

The layer is batch-only and has no serving path. **SOTA:** a point-in-time
feature store (Feast, Tecton). **Delta:** parquet plus a registry in S3 — at
one producer and two consumers, a served store is a component to operate for
a lookup a `read_parquet` already answers.
"""

from __future__ import annotations

from crucible.features.compute import LIQUIDITY_FLOOR_USD, build_features, read_features
from crucible.features.registry import (
    CATALOG,
    FEATURE_REGISTRY_SCHEMA_PATH,
    FEATURE_REGISTRY_SCHEMA_VERSION,
    UNIT_SUFFIXES,
    FeatureRegistryValidationError,
    FeatureSpec,
    feature_names,
    feature_version,
    load_registry_schema,
    registry_payload,
    validate_registry_payload,
)

#: The version the jobs write to unless told otherwise. Derived from the
#: catalogue, so it moves when the catalogue moves and cannot be forgotten.
DEFAULT_FEATURE_VERSION = feature_version(CATALOG)

__all__ = [
    "CATALOG",
    "DEFAULT_FEATURE_VERSION",
    "FEATURE_REGISTRY_SCHEMA_PATH",
    "FEATURE_REGISTRY_SCHEMA_VERSION",
    "LIQUIDITY_FLOOR_USD",
    "UNIT_SUFFIXES",
    "FeatureRegistryValidationError",
    "FeatureSpec",
    "build_features",
    "feature_names",
    "feature_version",
    "load_registry_schema",
    "read_features",
    "registry_payload",
    "validate_registry_payload",
]
