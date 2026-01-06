"""
Azure Blob Storage utilities for production deployment.
Handles reading CSV data and model files from Azure Blob Storage.

In production, set these environment variables:
- AZURE_STORAGE_CONNECTION_STRING: Your storage account connection string
- AZURE_STORAGE_CONTAINER: Container name (default: 'betting-data')

In local development, these are not needed - the app uses local file paths.
"""
import os
import io
from typing import Optional, BinaryIO
import pandas as pd

# Check if we're in production (Azure) or local development
IS_PRODUCTION = os.environ.get('AZURE_STORAGE_CONNECTION_STRING') is not None


def get_blob_service_client():
    """Get Azure Blob Service Client. Only call in production."""
    from azure.storage.blob import BlobServiceClient
    connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
    if not connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING not set")
    return BlobServiceClient.from_connection_string(connection_string)


def get_container_name() -> str:
    """Get the blob container name."""
    return os.environ.get('AZURE_STORAGE_CONTAINER', 'betting-data')


def download_blob_to_bytes(blob_path: str) -> bytes:
    """
    Download a blob to bytes.

    Args:
        blob_path: Path within the container (e.g., 'models/nhl_model.joblib')

    Returns:
        Blob contents as bytes
    """
    client = get_blob_service_client()
    container = get_container_name()
    blob_client = client.get_blob_client(container=container, blob=blob_path)
    return blob_client.download_blob().readall()


def download_blob_to_stream(blob_path: str) -> BinaryIO:
    """
    Download a blob to a BytesIO stream.

    Args:
        blob_path: Path within the container

    Returns:
        BytesIO stream with blob contents
    """
    data = download_blob_to_bytes(blob_path)
    return io.BytesIO(data)


def upload_blob_from_bytes(blob_path: str, data: bytes, overwrite: bool = True):
    """
    Upload bytes to a blob.

    Args:
        blob_path: Destination path within the container
        data: Bytes to upload
        overwrite: Whether to overwrite existing blob
    """
    client = get_blob_service_client()
    container = get_container_name()
    blob_client = client.get_blob_client(container=container, blob=blob_path)
    blob_client.upload_blob(data, overwrite=overwrite)


def upload_dataframe_as_csv(blob_path: str, df: pd.DataFrame, overwrite: bool = True):
    """
    Upload a pandas DataFrame as CSV to blob storage.

    Args:
        blob_path: Destination path (e.g., 'data/nhl_features.csv')
        df: DataFrame to upload
        overwrite: Whether to overwrite existing blob
    """
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode('utf-8')
    upload_blob_from_bytes(blob_path, csv_bytes, overwrite=overwrite)


def read_csv_from_blob(blob_path: str) -> pd.DataFrame:
    """
    Read a CSV file from blob storage into a DataFrame.

    Args:
        blob_path: Path within the container (e.g., 'data/nhl_features.csv')

    Returns:
        pandas DataFrame
    """
    stream = download_blob_to_stream(blob_path)
    return pd.read_csv(stream)


def load_joblib_model(blob_path: str):
    """
    Load a joblib model from blob storage.

    Args:
        blob_path: Path within the container (e.g., 'models/nhl_model.joblib')

    Returns:
        Loaded model object
    """
    import joblib
    stream = download_blob_to_stream(blob_path)
    return joblib.load(stream)


def blob_exists(blob_path: str) -> bool:
    """Check if a blob exists."""
    try:
        client = get_blob_service_client()
        container = get_container_name()
        blob_client = client.get_blob_client(container=container, blob=blob_path)
        return blob_client.exists()
    except Exception:
        return False


def list_blobs(prefix: str = '') -> list:
    """
    List blobs in the container with optional prefix filter.

    Args:
        prefix: Filter blobs by prefix (e.g., 'data/' or 'models/')

    Returns:
        List of blob names
    """
    client = get_blob_service_client()
    container = get_container_name()
    container_client = client.get_container_client(container)
    return [blob.name for blob in container_client.list_blobs(name_starts_with=prefix)]


# Convenience class for handling local vs production paths
class DataPath:
    """
    Helper class to resolve data paths for local development vs production.

    Usage:
        path = DataPath('nhl_model.joblib', local_path='/path/to/local/model.joblib', blob_path='models/nhl_model.joblib')

        if path.is_local:
            model = joblib.load(path.local)
        else:
            model = load_joblib_model(path.blob)
    """

    def __init__(self, name: str, local_path: str, blob_path: str):
        self.name = name
        self.local = local_path
        self.blob = blob_path

    @property
    def is_local(self) -> bool:
        """Check if we should use local path (development mode)."""
        return not IS_PRODUCTION

    @property
    def is_production(self) -> bool:
        """Check if we should use blob storage (production mode)."""
        return IS_PRODUCTION

    def exists(self) -> bool:
        """Check if the resource exists (local file or blob)."""
        if self.is_local:
            return os.path.exists(self.local)
        else:
            return blob_exists(self.blob)


# Pre-defined paths for common resources
class NHLPaths:
    """NHL resource paths for local and production."""

    BASE_LOCAL = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python-nhl-2026'))

    @classmethod
    def model(cls) -> DataPath:
        return DataPath(
            'nhl_model',
            local_path=os.path.join(cls.BASE_LOCAL, 'model_logit.joblib'),
            blob_path='models/nhl_model.joblib'
        )

    @classmethod
    def database(cls) -> DataPath:
        """For local: SQLite DB. For production: CSV features file."""
        return DataPath(
            'nhl_data',
            local_path=os.path.join(cls.BASE_LOCAL, 'nhl_scrape.sqlite'),
            blob_path='data/nhl_features.csv'
        )

    @classmethod
    def features_csv(cls) -> DataPath:
        """Latest features snapshot as CSV (production uses this instead of SQLite)."""
        return DataPath(
            'nhl_features',
            local_path=os.path.join(cls.BASE_LOCAL, 'nhl_features_latest.csv'),
            blob_path='data/nhl_features_latest.csv'
        )


class NBAPaths:
    """NBA resource paths for local and production."""

    BASE_LOCAL = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python-nba-2026'))

    @classmethod
    def model(cls) -> DataPath:
        return DataPath(
            'nba_model',
            local_path=os.path.join(cls.BASE_LOCAL, 'Models/homewin_logreg_final.joblib'),
            blob_path='models/nba_model.joblib'
        )

    @classmethod
    def schedule(cls) -> DataPath:
        return DataPath(
            'nba_schedule',
            local_path=os.path.join(cls.BASE_LOCAL, 'Data/nba-2025-UTC.csv'),
            blob_path='data/nba_schedule.csv'
        )
