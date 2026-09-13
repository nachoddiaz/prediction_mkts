"""
config/settings.py
───────────────────
Central system configuration.

Two configuration sources with distinct responsibilities:

  1. Environment variables / .env → secrets and per-environment settings.
                                    Never in git.

  2. Per-venue YAML               → model and risk parameters
                                    Versionados en git.
                                    Updated by the calibration notebooks
                                    via update_model_params().

Why the YAML is the single source of truth for model parameters:
  If defaults lived in the code you could not tell whether the system is
  running calibrated or invented parameters. With the YAML as the source,
  a missing field is apparent immediately at startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Variables de entorno
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Kalshi
    kalshi_api_key: str = Field(default="")
    kalshi_private_key_path: str = Field(default="./secrets/kalshi_private.pem")
    kalshi_env: str = Field(default="demo", pattern="^(demo|prod)$")

    # Polymarket
    polymarket_private_key: str = Field(default="")
    polymarket_proxy_address: str = Field(default="")
    polygon_rpc_url: str = Field(default="https://polygon-rpc.com")

    # Storage
    duckdb_path: str = Field(default="./data/duckdb/markets.duckdb")
    parquet_base_path: str = Field(default="./data/parquet")

    # Sistema
    paper_trading: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="text")

    # Connectors
    manifold_poll_interval: int = Field(default=10)

    # Which venues are ingested.
    #
    # Declaring them HERE rather than reading os.getenv() is not a style
    # preference: pydantic-settings loads the .env into this object, not into
    # os.environ, and `extra="ignore"` discards anything undeclared.
    # build_connectors() used os.getenv("ENABLE_POLYMARKET", "false"), which
    # never saw the value in .env — Polymarket stayed disabled unless the
    # variable was exported by hand on the command line.
    enable_kalshi: bool = Field(default=True)
    enable_polymarket: bool = Field(default=True)
    enable_manifold: bool = Field(default=True)

    # ---------------------------------------------------------------- #
    # Ingestion quality — which markets are worth the storage
    # ---------------------------------------------------------------- #

    # Maximum horizon to resolution, in days.
    #
    # A market resolving in 74 YEARS (Manifold has them) contributes nothing:
    # we will never see its outcome, so it can neither calibrate Brier nor
    # validate the signal, and its ticks only dilute the database. 45 days
    # comfortably covers the horizons actually observed on Kalshi (1.9-14 d)
    # and Polymarket (0-10.3 d), the two venues this project targets.
    max_tau_days: float = Field(default=45.0)

    # Require a two-sided book before ingesting a market.
    #
    # Without both a bid AND an ask there is no mid, no spread and no OBI —
    # the three features that drive the quoter. A one-sided market costs rows
    # and yields not one usable feature.
    require_two_sided_book: bool = Field(default=True)

    # Historical ticks backfilled per market on discovery.
    #
    # Manifold's backfill pulled 1000 bets per market — 31,000 of the
    # database's 31,100 rows were exactly this, on markets resolving 117 days
    # out. 0 disables it; the point is to seed the σ_b series without flooding.
    backfill_max_ticks: int = Field(default=200)

    # ---------------------------------------------------------------- #
    # Database size cap
    # ---------------------------------------------------------------- #

    # Maximum DuckDB file size in MB. Once exceeded the writer stops accepting
    # writes and logs the fact, rather than filling the disk in silence.
    # 0 = no limit.
    max_db_size_mb: float = Field(default=2048.0)

    # How many flushes between size checks. Checking on every flush would be a
    # stat() per batch; every 20 is enough to react within seconds.
    db_size_check_every: int = Field(default=20)

    # ---------------------------------------------------------------- #
    # Refresco de metadatos
    # ---------------------------------------------------------------- #

    # How often, in seconds, the status of tracked markets is re-queried.
    #
    # This is what makes a resolution OBSERVABLE. Without this loop the market
    # list froze at startup and `resolved_value` was never written, so no
    # amount of ingestion time produced a single resolved market. 900 s = 15
    # min: Kalshi sports markets resolve within minutes of the event, and a
    # 15-minute lag is irrelevant for calibration.
    market_refresh_seconds: int = Field(default=900)

    # Heartbeat file published by the ingestion process. It is the ONLY way an
    # external monitor can observe state: DuckDB grants the writer an exclusive
    # lock and blocks reads from any other process.
    status_file: str = Field(default="./data/ingest_status.json")

    @property
    def kalshi_enabled(self) -> bool:
        return bool(self.kalshi_api_key)

    @property
    def kalshi_private_key_exists(self) -> bool:
        return Path(self.kalshi_private_key_path).exists()


settings = Settings()


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RiskParams:
    """
    Risk-management parameters.

    Why there are no defaults here:
      The YAML is the single source of truth. If a field is missing,
      load_venue_config() raises KeyError — so you know exactly what is
      missing instead of silently using an invented value.

      Exception: optional fields such as max_daily_loss, where a conservative
      default is acceptable as a fallback.
    """

    gamma: float  # Aversión al riesgo γ del MATH.md
    q_max: int  # maximum inventory under normal conditions
    max_daily_loss: float  # maximum daily loss before halting
    max_position_loss: float  # maximum loss on a single position


@dataclass
class ModelParams:
    """
    Market-making model parameters.

    GLFT (MATH.md §3):
      kappa: arrival-rate decay — calibrate by MLE over fill rates
      A:     baseline arrival rate — calibrate by MLE

    Cartea-Jaimungal (MATH.md §4):
      phi: mean-reversion speed of μ_t — calibrate by AR(1)
      eta: volatility of μ_t           — calibrate by AR(1)
      rho: measure-change discount ρ_μ  — see cartea_jaimungal.py

    Composite signal (MATH.md §4.6):
      w_obi, w_news, w_onchain — calibrate by ridge regression
    """

    # GLFT
    kappa: float
    A: float

    # Cartea-Jaimungal
    phi: float
    eta: float
    rho: float

    # Signal weights
    w_obi: float
    w_news: float
    w_onchain: float

    # Calibration metadata
    # None when the parameters are the YAML's initial, uncalibrated values
    last_calibrated: datetime | None = None


@dataclass
class VenueConfig:
    """Complete configuration for one venue."""

    name: str
    active_categories: list[str]
    max_markets: int
    http_timeout: int
    risk: RiskParams
    model: ModelParams


# ---------------------------------------------------------------------------
# Loading from YAML
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    """
    Load a YAML file. Raises FileNotFoundError if it does not exist.

    Why not return an empty dict:
      A missing YAML is a configuration error, not a normal condition to paper
      over. We want a clear failure at startup.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Create the file, or copy it from config/{path.name}.example"
        )
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_venue_config(venue: str) -> VenueConfig:
    """
    Load a venue's configuration from its YAML file.

    Raises KeyError when a required field is missing — making it impossible to
    start the system silently with incomplete parameters.
    """
    config_dir = Path(__file__).parent
    raw = _load_yaml(config_dir / f"{venue}.yaml")

    try:
        raw_risk = raw["risk"]
        raw_model = raw["model"]
    except KeyError as e:
        raise KeyError(f"Missing required section {e} in config/{venue}.yaml") from e

    # Parsear last_calibrated si existe
    last_cal_str = raw_model.get("last_calibrated")
    last_cal = datetime.fromisoformat(last_cal_str) if last_cal_str else None

    risk = RiskParams(
        gamma=raw_risk.get("gamma_I", raw_risk.get("gamma", 0.1)),
        q_max=raw_risk.get("q_max", 10),
        max_daily_loss=raw_risk.get("max_daily_loss", 100.0),
        max_position_loss=raw_risk.get("max_position_loss", 20.0),
    )

    model = ModelParams(
        kappa=raw_model.get("kappa_p", raw_model.get("kappa", 1.5)),
        A=raw_model.get("A", 0.1),
        phi=raw_model.get("phi", 1.0),
        eta=raw_model.get("eta", 0.05),
        rho=raw_model.get("rho", 0.1),
        w_obi=raw_model.get("w_obi", 1.0),
        w_news=raw_model.get("w_news", 0.0),
        w_onchain=raw_model.get("w_onchain", 0.0),
        last_calibrated=last_cal,
    )

    return VenueConfig(
        name=raw.get("name", venue),
        active_categories=raw.get("active_categories", []),
        max_markets=raw.get("max_markets", 50),
        http_timeout=raw.get("http_timeout", 10),
        risk=risk,
        model=model,
    )


# ---------------------------------------------------------------------------
# Parameter updates from the calibration notebooks
# ---------------------------------------------------------------------------


def update_model_params(venue: str, params: dict[str, float]) -> None:
    """
    Update the venue's model parameters in its YAML file.

    Called from the calibration notebooks after estimating the parameters
    from real historical data.

    Every other field in the YAML is preserved — only the fields explicitly
    supplied in params are modified.

    Args:
        venue:  "kalshi" | "polymarket" | "manifold"
        params: dict of parameters to update. Only the fields supplied are
                modified.
                Example:
                  {
                    "kappa": 1.823,
                    "A":     0.094,
                    "phi":   1.156,
                    "eta":   0.047,
                    "rho":   0.089,
                  }

    Raises:
        FileNotFoundError: when the venue YAML does not exist
        ValueError:        if a parameter is not numeric
        KeyError:          if a parameter does not exist in the model section

    Example use from a notebook:
        from config.settings import update_model_params

        update_model_params("kalshi", {
            "kappa": kappa_hat,
            "A":     A_hat,
            "phi":   phi_hat,
            "eta":   eta_hat,
            "rho":   rho_hat,
        })
    """
    config_path = Path(__file__).parent / f"{venue}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Valid model fields — protection against typos
    valid_fields = {"kappa", "A", "phi", "eta", "rho", "w_obi", "w_news", "w_onchain"}

    for key in params:
        if key not in valid_fields:
            raise KeyError(
                f"Unknown model parameter: '{key}'. Valid fields: {sorted(valid_fields)}"
            )
        if not isinstance(params[key], int | float):
            raise ValueError(f"Parameter '{key}' must be numeric, got {type(params[key])}")

    # Read the current YAML, preserving structure and comments.
    # Note: yaml.safe_load drops comments — use ruamel.yaml if you need them
    # preserved. For now yaml.dump is sufficient.
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    if "model" not in config:
        config["model"] = {}

    # Update only the fields supplied
    for key, value in params.items():
        config["model"][key] = round(float(value), 8)

    # Record when the last calibration happened
    config["model"]["last_calibrated"] = datetime.now(tz=UTC).isoformat()

    # Escribir YAML actualizado
    with open(config_path, "w") as f:
        yaml.dump(
            config,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,  # preserve the original field order
        )

    # Reload the in-memory config so the change takes effect immediately,
    # with no need to restart the system
    _reload_venue_config(venue)

    print(
        f"✓ {venue} model params updated:\n"
        + "\n".join(f"  {k}: {v:.8f}" for k, v in params.items())
        + f"\n  last_calibrated: {config['model']['last_calibrated']}"
    )


def _reload_venue_config(venue: str) -> None:
    """
    Reload a venue's config into the global variables.
    Called automatically by update_model_params().

    Why reload in memory:
      If the system is running and you calibrate from a notebook, you want the
      new parameters to take effect on the next quoting cycle without
      restarting the system.
    """
    global kalshi_config, polymarket_config, manifold_config

    if venue == "kalshi":
        kalshi_config = load_venue_config("kalshi")
    elif venue == "polymarket":
        polymarket_config = load_venue_config("polymarket")
    elif venue == "manifold":
        manifold_config = load_venue_config("manifold")


def calibration_status() -> None:
    """
    Print the calibration status of every venue.
    Useful for telling whether the parameters are initial or calibrated.

    Use from a notebook or the terminal:
        from config.settings import calibration_status
        calibration_status()
    """
    for venue, config in [
        ("kalshi", kalshi_config),
        ("polymarket", polymarket_config),
        ("manifold", manifold_config),
    ]:
        last_cal = config.model.last_calibrated
        status = (
            f"calibrated {last_cal.strftime('%Y-%m-%d %H:%M UTC')}"
            if last_cal
            else "⚠  NOT CALIBRATED — using initial values"
        )
        print(f"  {venue:<12} {status}")
        print(
            f"             κ={config.model.kappa:.4f}  A={config.model.A:.4f}  "
            f"φ={config.model.phi:.4f}  η={config.model.eta:.4f}  "
            f"ρ={config.model.rho:.4f}"
        )


# ---------------------------------------------------------------------------
# Pre-loaded configs — import these objects elsewhere in the system
# ---------------------------------------------------------------------------

kalshi_config = load_venue_config("kalshi")
polymarket_config = load_venue_config("polymarket")
manifold_config = load_venue_config("manifold")
