import nflreadpy as nfl
from nflreadpy.config import update_config, CacheMode
from pathlib import Path

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


# CR opus: TTL defaults to 86400 seconds (1 day). During the offseason, player
# stats don't change — a longer TTL would avoid unnecessary re-fetches.
# During the season, 1 day may be too long if you need up-to-date weekly stats.
def configure_cache(cache_dir=None, ttl=86400):
    """Configure nflreadpy's filesystem cache."""
    cache_path = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_path.mkdir(parents=True, exist_ok=True)
    update_config(
        cache_mode=CacheMode.FILESYSTEM,
        cache_dir=cache_path,
        cache_duration=ttl,
    )


def clear_cache():
    """Clear the nflreadpy cache."""
    nfl.clear_cache()


# Auto-configure on import
configure_cache()
