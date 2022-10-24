"""Compression configuration for the application."""
from starlite.config.compression import CompressionConfig

config = CompressionConfig(backend="gzip")
"""Default compression config"""