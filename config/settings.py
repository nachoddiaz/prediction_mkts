"""
config/settings.py
───────────────────
Configuración central del sistema.

Dos fuentes de configuración con responsabilidades distintas:

  1. Variables de entorno / .env  → secretos y configuración por entorno
                                    Nunca en git.

  2. YAML de venue                → parámetros del modelo y riesgo
                                    Versionados en git.
                                    Actualizados por notebooks de calibración
                                    via update_model_params().

Por qué el YAML es la única fuente de verdad para parámetros del modelo:
  Si los defaults estuvieran en el código, no sabrías si el sistema
  está usando parámetros calibrados o inventados. Con el YAML como
  única fuente, si falta un campo lo sabes inmediatamente al arrancar.
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

    @property
    def kalshi_enabled(self) -> bool:
        return bool(self.kalshi_api_key)

    @property
    def kalshi_private_key_exists(self) -> bool:
        return Path(self.kalshi_private_key_path).exists()


settings = Settings()


# ---------------------------------------------------------------------------
# Dataclasses de parámetros
# ---------------------------------------------------------------------------


@dataclass
class RiskParams:
    """
    Parámetros de gestión de riesgo.

    Por qué no hay defaults aquí:
      El YAML es la única fuente de verdad. Si falta un campo en el YAML
      load_venue_config() lanza KeyError — así sabes exactamente
      qué falta en lugar de silenciosamente usar un valor inventado.

      Excepción: campos opcionales como max_daily_loss donde un default
      conservador es aceptable como fallback.
    """

    gamma: float  # Aversión al riesgo γ del MATH.md
    q_max: int  # Inventario máximo en condiciones normales
    max_daily_loss: float  # Pérdida diaria máxima antes de halt
    max_position_loss: float  # Pérdida máxima por posición


@dataclass
class ModelParams:
    """
    Parámetros del modelo de market making.

    GLFT (§3 del MATH.md):
      kappa: decay del arrival rate — calibrar por MLE sobre fill rates
      A:     baseline arrival rate  — calibrar por MLE

    Cartea-Jaimungal (§4 del MATH.md):
      phi: mean-reversion speed de μ_t — calibrar por AR(1)
      eta: volatilidad de μ_t         — calibrar por AR(1)
      rho: correlación Δp_t ↔ Δμ̂_t   — calibrar por sample correlation

    Señal compuesta (§4.6 del MATH.md):
      w_obi, w_news, w_onchain — calibrar por Ridge regression
    """

    # GLFT
    kappa: float
    A: float

    # Cartea-Jaimungal
    phi: float
    eta: float
    rho: float

    # Pesos de la señal
    w_obi: float
    w_news: float
    w_onchain: float

    # Metadata de calibración
    # None si los parámetros son los iniciales del YAML (sin calibrar)
    last_calibrated: datetime | None = None


@dataclass
class VenueConfig:
    """Configuración completa de una venue."""

    name: str
    active_categories: list[str]
    max_markets: int
    http_timeout: int
    risk: RiskParams
    model: ModelParams


# ---------------------------------------------------------------------------
# Carga desde YAML
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    """
    Carga un YAML. Lanza FileNotFoundError si no existe.

    Por qué no devolver dict vacío:
      Si el YAML no existe es un error de configuración, no una
      situación normal. Queremos un error claro al arrancar.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Crea el archivo o copia desde config/{path.name}.example"
        )
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_venue_config(venue: str) -> VenueConfig:
    """
    Carga la configuración de una venue desde su YAML.

    Lanza KeyError si falta un campo obligatorio — así es imposible
    arrancar el sistema con parámetros incompletos silenciosamente.
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
        gamma=raw_risk["gamma"],
        q_max=raw_risk["q_max"],
        max_daily_loss=raw_risk["max_daily_loss"],
        max_position_loss=raw_risk["max_position_loss"],
    )

    model = ModelParams(
        kappa=raw_model["kappa"],
        A=raw_model["A"],
        phi=raw_model["phi"],
        eta=raw_model["eta"],
        rho=raw_model["rho"],
        w_obi=raw_model["w_obi"],
        w_news=raw_model["w_news"],
        w_onchain=raw_model["w_onchain"],
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
# Actualización de parámetros desde notebooks de calibración
# ---------------------------------------------------------------------------


def update_model_params(venue: str, params: dict[str, float]) -> None:
    """
    Actualiza los parámetros del modelo en el YAML de la venue.

    Llamado desde los notebooks de calibración después de estimar
    los parámetros desde datos históricos reales.

    Preserva todos los demás campos del YAML — solo actualiza
    los campos explícitamente proporcionados en params.

    Args:
        venue:  "kalshi" | "polymarket" | "manifold"
        params: dict con los parámetros a actualizar.
                Solo los campos proporcionados se modifican.
                Ejemplo:
                  {
                    "kappa": 1.823,
                    "A":     0.094,
                    "phi":   1.156,
                    "eta":   0.047,
                    "rho":   0.089,
                  }

    Raises:
        FileNotFoundError: si el YAML del venue no existe
        ValueError:        si un parámetro no es numérico
        KeyError:          si un parámetro no existe en la sección model

    Ejemplo de uso desde notebook:
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

    # Campos válidos del modelo — protección contra typos
    valid_fields = {"kappa", "A", "phi", "eta", "rho", "w_obi", "w_news", "w_onchain"}

    for key in params:
        if key not in valid_fields:
            raise KeyError(
                f"Unknown model parameter: '{key}'. " f"Valid fields: {sorted(valid_fields)}"
            )
        if not isinstance(params[key], int | float):
            raise ValueError(f"Parameter '{key}' must be numeric, got {type(params[key])}")

    # Leer YAML actual preservando estructura y comentarios
    # Nota: yaml.safe_load pierde los comentarios — si quieres preservarlos
    # usa ruamel.yaml. Para ahora yaml.dump es suficiente.
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    if "model" not in config:
        config["model"] = {}

    # Actualizar solo los campos proporcionados
    for key, value in params.items():
        config["model"][key] = round(float(value), 8)

    # Registrar cuándo se calibró por última vez
    config["model"]["last_calibrated"] = datetime.now(tz=UTC).isoformat()

    # Escribir YAML actualizado
    with open(config_path, "w") as f:
        yaml.dump(
            config,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,  # preservar el orden original de los campos
        )

    # Recargar la config en memoria para que el cambio sea inmediato
    # sin necesidad de reiniciar el sistema
    _reload_venue_config(venue)

    print(
        f"✓ {venue} model params updated:\n"
        + "\n".join(f"  {k}: {v:.8f}" for k, v in params.items())
        + f"\n  last_calibrated: {config['model']['last_calibrated']}"
    )


def _reload_venue_config(venue: str) -> None:
    """
    Recarga la config de un venue en las variables globales.
    Llamado automáticamente por update_model_params().

    Por qué recargar en memoria:
      Si el sistema está corriendo y calibras en un notebook,
      quieres que los nuevos parámetros se usen en el siguiente
      ciclo de quoting sin reiniciar el sistema.
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
    Imprime el estado de calibración de todos los venues.
    Útil para saber si los parámetros son los iniciales o calibrados.

    Uso desde notebook o terminal:
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
# Configs pre-cargadas — importar estos objetos en el resto del sistema
# ---------------------------------------------------------------------------

kalshi_config = load_venue_config("kalshi")
polymarket_config = load_venue_config("polymarket")
manifold_config = load_venue_config("manifold")
