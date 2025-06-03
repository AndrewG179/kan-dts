REAL_DATASETS = ["sunspots", "daily_temp", "co2"]

SYN_SPECS = [
    # (kind, noise_std)   – list is easier to iterate over
    ("trend_season", 0.0),
    ("trend_season", 0.05),
    ("trend_season", 0.10),
    ("trend_only",   0.05),
    ("season_only",  0.05),
]