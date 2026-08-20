"""Readable names for fields that only legacy-rejection tests may construct."""

SHARED_CAUSE_FIELDS = (
    "ss_dominated_undefined_frac",
    "oo_dominated_undefined_frac",
    "mixed_undefined_frac",
)
RETIRED_AGGREGATE_FIELD = "undefined_frac"
RETIRED_FLAGSHIP_AGGREGATE_FIELD = "croma_undefined_frac"
RETIRED_FLAGSHIP_SUPPORT_FIELD = "croma_support"


def retired_metric_prefixed_fields() -> set[str]:
    """Return the RI/MaRI aliases rejected by the shared output schema."""
    return {
        f"{metric}_{field}"
        for metric in ("ri", "mari")
        for field in (RETIRED_AGGREGATE_FIELD, *SHARED_CAUSE_FIELDS)
    }
