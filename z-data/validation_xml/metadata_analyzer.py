"""
OData $metadata analyzer for dynamic query validation limits.

Parses the OData $metadata XML and produces per-entity ``EntityProfile``
objects that tell the validator how large/expensive a given entity is, so
that $top limits and cost scores can be adjusted per entity rather than
using flat config-wide thresholds.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


# OData XML namespaces
_NS = {
    "edmx": "http://docs.oasis-open.org/odata/ns/edmx",
    "edm": "http://docs.oasis-open.org/odata/ns/edm",
}


class EntitySize(Enum):
    """
    Coarse classification of an entity's expected data volume.

    Used to pick appropriate $top thresholds and cost multipliers.
    """
    SMALL = "small"          # Reference / lookup tables  (<15 props)
    MEDIUM = "medium"        # Moderate entities          (15–40 props)
    LARGE = "large"          # Wide entities              (>40 props)
    TRANSACTION = "transaction"  # Time-keyed ledger tables (multi-key + temporal)


# Per-size default limits: (warn_top_threshold, max_top_limit)
_SIZE_LIMITS: Dict[EntitySize, tuple[int, int]] = {
    EntitySize.SMALL:       (1000, 5000),
    EntitySize.MEDIUM:      (500,  1000),
    EntitySize.LARGE:       (200,  500),
    EntitySize.TRANSACTION: (100,  500),
}


@dataclass(frozen=True)
class EntityProfile:
    """
    Derived metadata for a single OData entity set.

    Attributes
    ----------
    name : str
        EntitySet name as it appears in the query (e.g. ``"BankTrans"``).
    property_count : int
        Number of scalar properties on the underlying EntityType.
    key_count : int
        Number of key properties (composite key → higher cardinality).
    nav_property_count : int
        Number of navigation properties (potential $expand targets).
    has_temporal_key : bool
        True when at least one key property name contains a date/time hint.
    size : EntitySize
        Coarse size classification.
    warn_top_threshold : int
        $top value above which a WARNING is issued.
    max_top_limit : int
        $top value above which the query is BLOCKED (CRITICAL).
    width_factor : float
        Relative row width vs. a 25-property baseline (0.5–2.0).
        Used to scale the $top contribution to the cost score.
    """

    name: str
    property_count: int
    key_count: int
    nav_property_count: int
    has_temporal_key: bool
    size: EntitySize
    warn_top_threshold: int
    max_top_limit: int
    width_factor: float

    @property
    def is_transaction_table(self) -> bool:
        return self.size == EntitySize.TRANSACTION

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "property_count": self.property_count,
            "key_count": self.key_count,
            "nav_property_count": self.nav_property_count,
            "has_temporal_key": self.has_temporal_key,
            "size": self.size.value,
            "warn_top_threshold": self.warn_top_threshold,
            "max_top_limit": self.max_top_limit,
            "width_factor": self.width_factor,
        }


class MetadataAnalyzer:
    """
    Parses an OData ``$metadata`` XML document and returns per-entity profiles.

    Usage
    -----
    .. code-block:: python

        with open("$metadata.xml") as f:
            xml_content = f.read()

        profiles = MetadataAnalyzer.parse(xml_content)
        profile = profiles.get("BankTrans")
    """

    @staticmethod
    def parse(xml_content: str) -> Dict[str, EntityProfile]:
        """
        Parse ``$metadata`` XML and return a mapping of EntitySet name → profile.

        The method gracefully strips any leading browser preamble text that
        some servers/browsers prepend before the ``<edmx:Edmx ...>`` element.

        Parameters
        ----------
        xml_content : str
            Raw content of the OData ``$metadata`` response.

        Returns
        -------
        Dict[str, EntityProfile]
            Map from EntitySet name (as used in queries) to its profile.
            Returns an empty dict if the XML cannot be parsed.
        """
        # Strip any browser preamble before the actual XML
        xml_start = xml_content.find("<edmx:Edmx")
        if xml_start == -1:
            xml_start = xml_content.find("<?xml")
        if xml_start > 0:
            xml_content = xml_content[xml_start:]

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError:
            return {}

        schema = root.find(".//edm:Schema", _NS)
        if schema is None:
            return {}

        # Step 1 — build EntityType name → raw stats
        et_stats: Dict[str, dict] = {}
        for et in schema.findall("edm:EntityType", _NS):
            et_name = et.get("Name", "")
            props = et.findall("edm:Property", _NS)
            keys = et.findall(".//edm:PropertyRef", _NS)
            nav_props = et.findall("edm:NavigationProperty", _NS)
            key_names = [k.get("Name", "") for k in keys]

            temporal_indicators = ("date", "time", "period", "year", "month")
            has_temporal_key = any(
                any(ind in k.lower() for ind in temporal_indicators)
                for k in key_names
            )

            et_stats[et_name] = {
                "prop_count": len(props),
                "key_count": len(key_names),
                "nav_count": len(nav_props),
                "temporal_key": has_temporal_key,
            }

        # Step 2 — map EntitySet → EntityType, then build profiles
        container = schema.find("edm:EntityContainer", _NS)
        if container is None:
            return {}

        profiles: Dict[str, EntityProfile] = {}
        for es in container.findall("edm:EntitySet", _NS):
            es_name = es.get("Name", "")
            et_type = es.get("EntityType", "").split(".")[-1]  # strip namespace
            stats = et_stats.get(et_type)
            if stats is None:
                continue

            size = MetadataAnalyzer._classify(
                stats["prop_count"], stats["key_count"], stats["temporal_key"]
            )
            warn_threshold, max_limit = _SIZE_LIMITS[size]
            width_factor = MetadataAnalyzer._width_factor(stats["prop_count"])

            profiles[es_name] = EntityProfile(
                name=es_name,
                property_count=stats["prop_count"],
                key_count=stats["key_count"],
                nav_property_count=stats["nav_count"],
                has_temporal_key=stats["temporal_key"],
                size=size,
                warn_top_threshold=warn_threshold,
                max_top_limit=max_limit,
                width_factor=width_factor,
            )

        return profiles

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _classify(prop_count: int, key_count: int, has_temporal_key: bool) -> EntitySize:
        """Classify an entity into a size bucket."""
        # Composite temporal key → financial ledger / time-series table
        if has_temporal_key and key_count >= 3:
            return EntitySize.TRANSACTION
        if prop_count > 40:
            return EntitySize.LARGE
        if prop_count >= 15:
            return EntitySize.MEDIUM
        return EntitySize.SMALL

    @staticmethod
    def _width_factor(prop_count: int) -> float:
        """
        Return a multiplier reflecting how wide each row is.

        Baseline is 25 properties = 1.0x.  Clamped to [0.5, 2.0].
        """
        return round(min(2.0, max(0.5, prop_count / 25.0)), 2)
