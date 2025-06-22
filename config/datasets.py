REAL_DATASETS = ["sunspots", "daily_temp", "co2"]

# Original synthetic datasets plus additional season-only tests
SYN_SPECS = [
    # Original datasets
    ("trend_season", 0.05),
    ("trend_season", 0.10),
    ("trend_only",   0.05),
    ("trend_only",   0.1),
    ("season_only",  0.0),
    ("season_only",  0.05),
    ("season_only",  0.1),
    ("season_only",  0.2),
]

# Comprehensive noise testing (separate from main comparison)
NOISE_ANALYSIS_SPECS = [
    ("trend_only", 0.0),
    ("trend_only", 0.05),
    ("trend_only", 0.1),
    ("trend_only", 0.2),
    ("trend_only", 0.5),
    ("trend_only", 1.0),
    ("season_only", 0.0),
    ("season_only", 0.05),
    ("season_only", 0.1),
    ("season_only", 0.2),
    ("season_only", 0.5),
    ("season_only", 1.0),
    ("trend_season", 0.0),
    ("trend_season", 0.05),
    ("trend_season", 0.1),
    ("trend_season", 0.2),
    ("trend_season", 0.5),
    ("trend_season", 1.0),
]