from dotenv import load_dotenv
import os
from pathlib import Path

class Config:
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parent
        self.load_environment()
        
    def load_environment(self):
        """Load environment variables from .env file"""
        env_path = self.root_dir / '.env'
        
        # Try multiple possible locations for .env
        possible_paths = [
            env_path,
            Path.cwd() / '.env',
            Path.home() / '.env'
        ]
        
        for path in possible_paths:
            if path.exists():
                load_dotenv(path)
                break
        else:
            raise FileNotFoundError(f".env file not found in any of: {possible_paths}")
        
        self.fred_api_key = os.getenv('FRED_API_KEY')
        if not self.fred_api_key:
            raise ValueError("FRED_API_KEY not found in .env file")

config = Config()