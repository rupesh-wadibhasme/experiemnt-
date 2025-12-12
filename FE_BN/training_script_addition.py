def stats_to_jsonable(stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Convert stats_train dict to JSON-safe structure:
      { combo_str: {median_log: float, iqr_log: float, mad_inr: float, count: int} }
    """
    out: Dict[str, Dict[str, float]] = {}
    for combo, d in stats.items():
        out[str(combo)] = {
            "median_log": float(d.get("median_log", 0.0)),
            "iqr_log": float(d.get("iqr_log", 1.0)),
            "mad_inr": float(d.get("mad_inr", 1.0)),
            "count": int(d.get("count", 0)),
        }
    return out


def thresholds_to_jsonable(thr_global: float,
                           thr_per_combo: Dict[str, float]) -> Dict[str, Any]:
    """
    Convert thresholds to JSON-safe structure:
      {
        "thr_global": float,
        "thr_per_combo": { combo_str: float, ... }
      }
    """
    return {
        "thr_global": float(thr_global),
        "thr_per_combo": {str(c): float(v) for c, v in thr_per_combo.items()},
    }


# ====

# Save per-combo stats for inference
os.makedirs(OUT_DIR, exist_ok=True)
combo_stats_path = os.path.join(OUT_DIR, COMBO_STATS_JSON)
with open(combo_stats_path, "w") as f:
    json.dump(stats_to_jsonable(stats_train), f, indent=2)

#============

    # Save thresholds for inference
    thr_obj = thresholds_to_jsonable(thr_global, thr_per_combo)
    thresholds_path = os.path.join(OUT_DIR, THRESHOLDS_JSON)
    with open(thresholds_path, "w") as f:
        json.dump(thr_obj, f, indent=2)


    # Save per-combo stats for inference
    os.makedirs(OUT_DIR, exist_ok=True)
    combo_stats_path = os.path.join(OUT_DIR, COMBO_STATS_JSON)
    with open(combo_stats_path, "w") as f:
        json.dump(stats_to_jsonable(stats_train), f, indent=2)


#=======

    # Save thresholds for inference
    thr_obj = thresholds_to_jsonable(thr_global, thr_per_combo)
    thresholds_path = os.path.join(OUT_DIR, THRESHOLDS_JSON)
    with open(thresholds_path, "w") as f:
        json.dump(thr_obj, f, indent=2)


