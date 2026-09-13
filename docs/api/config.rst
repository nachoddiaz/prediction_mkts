config
======

Central configuration for the system.  Settings are loaded from two
independent sources:

1. **Environment variables / .env** — secrets and per-environment configuration.
   Never committed to git.
2. **Venue YAML files** — model and risk parameters versioned in git and
   updated by calibration notebooks via :func:`~config.settings.update_model_params`.

.. automodule:: config.settings
   :members:
   :undoc-members:
   :show-inheritance:
