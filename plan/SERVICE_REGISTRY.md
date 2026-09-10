# Gnosis Service Registry

This document describes the service discovery pattern for the gnosis stack.

## Overview

Rather than hardcoding service URLs and ports in test scripts and configuration files, gnosis services use a centralized registry pattern to eliminate brittle dependencies and improve development workflow.

## Configuration File

The `gnosis_services.json` file defines service endpoints for different environments:

```json
{
  "development": {
    "gnosis-ahp": {
      "url": "http://localhost:6793",
      "description": "Authentication and HTTP Proxy service"
    },
    "gnosis-crawl": {
      "url": "http://localhost:6792", 
      "description": "Web crawling and markdown generation service"
    }
  },
  "test": {
    // Same as development for testing
  },
  "production": {
    "gnosis-ahp": {
      "url": "${GNOSIS_AHP_URL}",
      "description": "Uses environment variables"
    }
  }
}
```

## Registry Usage

### In Test Scripts

```python
from gnosis_registry import registry

# Get service URLs
ahp_url = registry.ahp_url
crawl_url = registry.crawl_url

# Or use the full service info
service_info = registry.get_service_info("gnosis-ahp")
```

### Command Line Override

Test scripts still support URL overrides for flexibility:

```bash
# Use registry defaults
python test_simple.py --service-token <token>

# Override specific service
python test_simple.py --service-token <token> --crawl-url http://localhost:8080
```

## Port Mapping

| Service | Development Port | Production |
|---------|------------------|------------|
| gnosis-ahp | 6793 | Environment variable |
| gnosis-crawl | 6792 | Environment variable |
| gnosis-ocr | 6794 | Environment variable |

## Benefits

1. **No hardcoded ports**: Eliminates "magic numbers" in test scripts
2. **Environment awareness**: Different configs for dev/test/prod
3. **Single source of truth**: All service locations in one file
4. **Flexibility**: Command line overrides still available
5. **Error reduction**: Prevents "port 5005 vs 6793" type errors

## Future Enhancements

- Auto-discovery via service mesh or DNS
- Health checking and failover URLs
- Load balancing for production deployments
- Integration with Docker Compose service names