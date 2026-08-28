"""
Configuration Management for ML Model Serving API

This module handles all configuration for the application:
- Environment variables
- Model settings
- Server settings
- Monitoring settings
- Feature flags

Students should implement configuration loading and validation.
"""

import os
from typing import Optional, List
from pydantic import BaseSettings, Field, validator
from pathlib import Path


# ==============================================================================
# TODO: Application Configuration
# ==============================================================================
# Implement configuration class using Pydantic BaseSettings
# This provides:
# - Automatic loading from environment variables
# - Type validation
# - Default values
# - Documentation

class Settings(BaseSettings):
    """
    TODO: Application settings loaded from environment variables

    This class uses Pydantic BaseSettings to automatically load
    configuration from environment variables with type validation.

    Steps to implement:
    1. Define all configuration fields with types
    2. Add default values where appropriate
    3. Add validators for complex validation
    4. Add documentation for each field
    5. Configure environment variable prefix

    Example usage:
        settings = Settings()
        print(settings.model_name)  # Reads MODEL_NAME env var
        print(settings.port)        # Reads PORT env var or uses default

    Environment variables can be set:
        export MODEL_NAME=resnet50
        export PORT=8000
        export LOG_LEVEL=info

    Or in .env file:
        MODEL_NAME=resnet50
        PORT=8000
        LOG_LEVEL=info
    """

    # TODO: Application Settings
    app_name: str = Field(
        default="ML Model Serving API",
        description="Application name"
    )
    app_version: str = Field(
        default="1.0.0",
        description="Application version"
    )
    environment: str = Field(
        default="development",
        description="Environment (development, staging, production)"
    )

    # TODO: Server Settings
    host: str = Field(
        default="0.0.0.0",
        description="Server host address"
    )
    port: int = Field(
        default=8000,
        description="Server port"
    )
    workers: int = Field(
        default=1,
        description="Number of worker processes"
    )
    reload: bool = Field(
        default=False,
        description="Auto-reload on code changes (development only)"
    )

    # TODO: Model Settings
    model_name: str = Field(
        default="resnet18",
        description="Model name to load (resnet18, resnet50, mobilenet_v2)"
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Path to model weights file (if using custom model)"
    )
    device: str = Field(
        default="cpu",
        description="Device to use (cpu, cuda, cuda:0, cuda:1)"
    )
    batch_size: int = Field(
        default=1,
        description="Batch size for inference"
    )
    warmup_iterations: int = Field(
        default=10,
        description="Number of warmup iterations on startup"
    )

    # TODO: Inference Settings
    max_image_size_mb: int = Field(
        default=10,
        description="Maximum image file size in MB"
    )
    allowed_image_types: List[str] = Field(
        default=["image/jpeg", "image/png", "image/jpg"],
        description="Allowed image MIME types"
    )
    default_top_k: int = Field(
        default=5,
        description="Default number of top predictions to return"
    )
    confidence_threshold: float = Field(
        default=0.0,
        description="Minimum confidence threshold for predictions"
    )

    # TODO: Logging Settings
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    log_format: str = Field(
        default="json",
        description="Log format (json, text)"
    )

    # TODO: Monitoring Settings
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )
    metrics_port: int = Field(
        default=8000,
        description="Port for metrics endpoint (usually same as app port)"
    )

    # TODO: Caching Settings
    enable_cache: bool = Field(
        default=False,
        description="Enable prediction caching"
    )
    cache_size: int = Field(
        default=1000,
        description="Maximum number of cached predictions"
    )
    cache_ttl_seconds: int = Field(
        default=300,
        description="Cache time-to-live in seconds"
    )

    # TODO: Rate Limiting Settings
    enable_rate_limit: bool = Field(
        default=False,
        description="Enable rate limiting"
    )
    rate_limit_requests: int = Field(
        default=100,
        description="Maximum requests per time window"
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        description="Rate limit time window in seconds"
    )

    # TODO: CORS Settings (Cross-Origin Resource Sharing)
    enable_cors: bool = Field(
        default=True,
        description="Enable CORS"
    )
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )

    # TODO: Health Check Settings
    health_check_interval_seconds: int = Field(
        default=30,
        description="Interval between health checks"
    )

    # ==============================================================================
    # TODO: Validators
    # ==============================================================================
    # Add custom validators for complex validation logic

    @validator("environment")
    def validate_environment(cls, v):
        """
        TODO: Validate environment value

        Steps to implement:
        1. Check if environment is one of allowed values
        2. Raise ValueError if invalid
        3. Return value if valid

        Allowed values: development, staging, production

        Example:
            allowed = ["development", "staging", "production"]
            if v not in allowed:
                raise ValueError(f"Environment must be one of: {allowed}")
            return v
        """
        # TODO: Implement environment validation
        pass

    @validator("port", "metrics_port")
    def validate_port(cls, v):
        """
        TODO: Validate port number

        Steps to implement:
        1. Check if port is in valid range (1-65535)
        2. Raise ValueError if invalid
        3. Return value if valid

        Example:
            if not 1 <= v <= 65535:
                raise ValueError("Port must be between 1 and 65535")
            return v
        """
        # TODO: Implement port validation
        pass

    @validator("model_name")
    def validate_model_name(cls, v):
        """
        TODO: Validate model name

        Steps to implement:
        1. Check if model name is supported
        2. Raise ValueError if unsupported
        3. Return value if valid

        Supported models: resnet18, resnet50, mobilenet_v2

        Example:
            supported = ["resnet18", "resnet50", "mobilenet_v2"]
            if v not in supported:
                raise ValueError(f"Model must be one of: {supported}")
            return v
        """
        # TODO: Implement model name validation
        pass

    @validator("device")
    def validate_device(cls, v):
        """
        TODO: Validate device string

        Steps to implement:
        1. Check if device format is valid (cpu, cuda, cuda:0, etc.)
        2. If cuda, verify CUDA is available
        3. Raise ValueError if invalid
        4. Return value if valid

        Valid formats:
        - "cpu"
        - "cuda" (uses default GPU)
        - "cuda:0", "cuda:1", etc. (specific GPU)

        Example:
            import torch

            if v.startswith("cuda"):
                if not torch.cuda.is_available():
                    raise ValueError("CUDA device specified but CUDA not available")

            if ":" in v:
                device_type, device_id = v.split(":")
                if device_type != "cuda":
                    raise ValueError("Only 'cuda:N' format supported")
                try:
                    device_id = int(device_id)
                except ValueError:
                    raise ValueError("Device ID must be integer")

            return v
        """
        # TODO: Implement device validation
        pass

    @validator("log_level")
    def validate_log_level(cls, v):
        """
        TODO: Validate log level

        Steps to implement:
        1. Convert to uppercase
        2. Check if valid log level
        3. Raise ValueError if invalid
        4. Return uppercase value

        Valid levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

        Example:
            v = v.upper()
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if v not in valid_levels:
                raise ValueError(f"Log level must be one of: {valid_levels}")
            return v
        """
        # TODO: Implement log level validation
        pass

    @validator("confidence_threshold")
    def validate_confidence_threshold(cls, v):
        """
        TODO: Validate confidence threshold

        Steps to implement:
        1. Check if value is between 0.0 and 1.0
        2. Raise ValueError if out of range
        3. Return value if valid

        Example:
            if not 0.0 <= v <= 1.0:
                raise ValueError("Confidence threshold must be between 0.0 and 1.0")
            return v
        """
        # TODO: Implement confidence threshold validation
        pass

    # ==============================================================================
    # TODO: Configuration Methods
    # ==============================================================================
    # Add helper methods to work with configuration

    def is_production(self) -> bool:
        """
        TODO: Check if running in production

        Returns:
            True if environment is "production"

        Example usage:
            if settings.is_production():
                # Enable production features
                pass
        """
        # TODO: Implement production check
        pass

    def is_development(self) -> bool:
        """
        TODO: Check if running in development

        Returns:
            True if environment is "development"

        Example usage:
            if settings.is_development():
                # Enable development features (debug, auto-reload)
                pass
        """
        # TODO: Implement development check
        pass

    def get_device(self):
        """
        TODO: Get PyTorch device object

        Steps to implement:
        1. Import torch
        2. Create device from device string
        3. Return torch.device object

        Returns:
            torch.device object

        Example usage:
            device = settings.get_device()
            model.to(device)

        Implementation:
            import torch
            return torch.device(self.device)
        """
        # TODO: Implement device getter
        pass

    def get_model_path(self) -> Path:
        """
        TODO: Get full path to model file

        Steps to implement:
        1. If model_path is set, return it
        2. Otherwise, construct path from model_name
        3. Check if path exists
        4. Return Path object

        Returns:
            Path object to model file

        Example usage:
            model_path = settings.get_model_path()
            if model_path.exists():
                # Load model from path
                pass

        Implementation:
            if self.model_path:
                return Path(self.model_path)

            # Default path: models/{model_name}.pth
            default_path = Path("models") / f"{self.model_name}.pth"
            return default_path
        """
        # TODO: Implement model path getter
        pass

    def to_dict(self) -> dict:
        """
        TODO: Convert settings to dictionary

        Returns:
            Dictionary of all settings

        Example usage:
            config_dict = settings.to_dict()
            logger.info(f"Configuration: {config_dict}")

        Implementation:
            return self.dict()
        """
        # TODO: Implement to_dict
        pass

    class Config:
        """
        TODO: Pydantic configuration

        Configure how Pydantic loads settings:
        - env_file: Load from .env file
        - env_file_encoding: File encoding
        - case_sensitive: Whether env var names are case-sensitive
        - env_prefix: Prefix for environment variables

        Example .env file:
            MODEL_NAME=resnet50
            PORT=8000
            LOG_LEVEL=debug
            DEVICE=cuda
        """
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        # env_prefix = "APP_"  # If you want APP_MODEL_NAME instead of MODEL_NAME


# ==============================================================================
# TODO: Configuration Loading Functions
# ==============================================================================
# Implement functions to load and manage configuration

def load_settings() -> Settings:
    """
    TODO: Load application settings

    Steps to implement:
    1. Create Settings instance (automatically loads from env)
    2. Validate all settings
    3. Log configuration (mask sensitive values)
    4. Return settings object

    Returns:
        Configured Settings object

    Example usage:
        settings = load_settings()
        print(f"Starting server on port {settings.port}")
        print(f"Using model: {settings.model_name}")
        print(f"Device: {settings.device}")

    Implementation:
        try:
            settings = Settings()

            # Log configuration (be careful with sensitive data)
            print(f"Loaded configuration:")
            print(f"  Environment: {settings.environment}")
            print(f"  Model: {settings.model_name}")
            print(f"  Device: {settings.device}")
            print(f"  Port: {settings.port}")

            return settings

        except Exception as e:
            print(f"Failed to load configuration: {e}")
            raise
    """
    # TODO: Implement settings loading
    pass


def get_settings() -> Settings:
    """
    TODO: Get cached settings instance (singleton pattern)

    This ensures we only load settings once per application lifecycle.

    Steps to implement:
    1. Check if settings already loaded
    2. If not, load settings
    3. Cache and return settings

    Returns:
        Cached Settings object

    Example usage:
        # In api.py
        from config import get_settings

        settings = get_settings()
        # Use settings throughout the application

    Implementation (using functools.lru_cache):
        from functools import lru_cache

        @lru_cache()
        def get_settings():
            return load_settings()
    """
    # TODO: Implement cached settings getter
    pass


def override_settings(**kwargs) -> Settings:
    """
    TODO: Create settings with overrides (useful for testing)

    Steps to implement:
    1. Load base settings
    2. Override with provided kwargs
    3. Return new settings object

    Args:
        **kwargs: Settings to override

    Returns:
        Settings object with overrides

    Example usage:
        # In tests
        test_settings = override_settings(
            model_name="resnet18",
            device="cpu",
            enable_cache=False
        )

    Implementation:
        settings_dict = load_settings().dict()
        settings_dict.update(kwargs)
        return Settings(**settings_dict)
    """
    # TODO: Implement settings override
    pass


# ==============================================================================
# TODO: Environment-Specific Configuration
# ==============================================================================
# Implement functions to handle environment-specific settings

def get_development_settings() -> Settings:
    """
    TODO: Get development-specific settings

    Returns settings optimized for development:
    - Debug logging
    - Auto-reload enabled
    - Smaller batch sizes
    - CPU device (unless GPU specified)

    Returns:
        Development Settings object

    Example usage:
        if os.getenv("ENV") == "development":
            settings = get_development_settings()
    """
    # TODO: Implement development settings
    pass


def get_production_settings() -> Settings:
    """
    TODO: Get production-specific settings

    Returns settings optimized for production:
    - Info logging
    - Auto-reload disabled
    - Larger batch sizes
    - GPU device if available
    - Metrics enabled
    - Rate limiting enabled

    Returns:
        Production Settings object

    Example usage:
        if os.getenv("ENV") == "production":
            settings = get_production_settings()
    """
    # TODO: Implement production settings
    pass


# ==============================================================================
# TODO: Configuration Validation
# ==============================================================================
# Implement comprehensive configuration validation

def validate_configuration(settings: Settings) -> List[str]:
    """
    TODO: Validate entire configuration

    Performs comprehensive validation beyond field-level checks:
    - Check if model files exist
    - Verify GPU availability if GPU specified
    - Check port availability
    - Verify required directories exist

    Args:
        settings: Settings object to validate

    Returns:
        List of validation errors (empty if valid)

    Example usage:
        settings = load_settings()
        errors = validate_configuration(settings)
        if errors:
            for error in errors:
                print(f"Configuration error: {error}")
            sys.exit(1)

    Validations to implement:
        - Model file exists (if model_path specified)
        - CUDA available (if cuda device specified)
        - Port not in use
        - Directories exist (models, logs, cache)
        - Write permissions
    """
    # TODO: Implement configuration validation
    pass


# ==============================================================================
# Example Usage
# ==============================================================================
# After implementing the above, you can use configuration like this:
#
# # In api.py
# from config import get_settings
#
# settings = get_settings()
#
# app = FastAPI(
#     title=settings.app_name,
#     version=settings.app_version
# )
#
# @app.on_event("startup")
# async def startup():
#     model = load_model(settings.model_name)
#     model.to(settings.get_device())
#
# if __name__ == "__main__":
#     uvicorn.run(
#         app,
#         host=settings.host,
#         port=settings.port,
#         workers=settings.workers
#     )
# ==============================================================================

# ==============================================================================
# Testing Configuration
# ==============================================================================
# Test configuration with:
#
# 1. Unit tests
#     def test_load_settings():
#         settings = load_settings()
#         assert settings.port > 0
#         assert settings.model_name in ["resnet18", "resnet50"]
#
# 2. Integration tests with different .env files
# 3. Validation tests for each validator
# 4. Override tests for testing scenarios
# ==============================================================================
