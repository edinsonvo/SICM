"""Configuration settings module."""


class Settings:
    """Application settings."""
    
    def __init__(self, app_name: str = "SICM Core"):
        """Initialize settings with default values.
        
        Args:
            app_name: Application name, defaults to "SICM Core"
        """
        self.app_name = app_name
