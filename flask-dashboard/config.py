"""
Flask Application Configuration

Environment Variables:
- FLASK_ENV: 'development' or 'production'
- SECRET_KEY: Secret key for sessions (required in production)
- AZURE_STORAGE_CONNECTION_STRING: Azure Blob Storage connection string (production only)
- AZURE_STORAGE_CONTAINER: Blob container name (default: 'betting-data')
"""
import os


def is_production() -> bool:
    """Check if running in production mode."""
    return os.environ.get('AZURE_STORAGE_CONNECTION_STRING') is not None


class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

    # Flask settings
    DEBUG = False
    TESTING = False

    # Environment detection
    IS_PRODUCTION = is_production()

    # Azure settings
    AZURE_STORAGE_CONNECTION_STRING = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
    AZURE_STORAGE_CONTAINER = os.environ.get('AZURE_STORAGE_CONTAINER', 'betting-data')

    # Model paths (relative to flask-dashboard directory, used in local development)
    MODEL_PATHS = {
        'nba': '../python-nba-2026',
        'nhl': '../python-nhl-2026',
        'mlb': '../python-mlb-2026'
    }

    # Azure Blob paths (used in production)
    BLOB_PATHS = {
        'nhl_model': 'models/nhl_model.joblib',
        'nhl_features': 'data/nhl_features_latest.csv',
        'nba_model': 'models/nba_model.joblib',
        'nba_schedule': 'data/nba_schedule.csv',
    }


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration for Azure deployment."""
    DEBUG = False
    # Azure App Service will set this
    SECRET_KEY = os.environ.get('SECRET_KEY')


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True


# Config selector
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
