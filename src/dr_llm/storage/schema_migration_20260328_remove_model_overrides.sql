UPDATE provider_models_current
SET source_quality = 'live'
WHERE source_quality = 'overlay';

DROP TABLE IF EXISTS provider_model_overrides;
