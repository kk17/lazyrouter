"""CLI entry point for LazyRouter server."""

import argparse
import json
import os
import re
from typing import Any, Dict

import httpx
import uvicorn

from .config import load_config
from .server import create_app

_CONFIG_ENV_VAR = "LAZYROUTER_CONFIG_PATH"
_ENV_FILE_ENV_VAR = "LAZYROUTER_ENV_FILE"


def _fetch_provider_models(provider_name: str, provider_cfg: Any) -> Dict[str, Any]:
    """Fetch model metadata from a provider's model list API.

    Returns a dict mapping provider model ID -> metadata dict.
    Silently returns {} on any error.
    """
    api_style: str = getattr(provider_cfg, "api_style", "openai")
    api_key: str = getattr(provider_cfg, "api_key", "") or ""
    base_url: str = (getattr(provider_cfg, "base_url", None) or "").rstrip("/")

    _versioned = bool(re.search(r"/v\d+$", base_url))

    headers: Dict[str, str] = {}
    url: str = ""

    if api_style == "anthropic":
        url = f"{base_url}/models" if _versioned else f"{base_url}/v1/models"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
    elif api_style == "github-copilot":
        url = f"{base_url}/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Copilot-Integration-Id": "vscode-chat",
            "Editor-Version": "vscode/1.99.0",
        }
    else:
        # openai / openai-completions / gemini / etc.
        url = f"{base_url}/models" if _versioned else f"{base_url}/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        print(f"  Warning: failed to fetch models from {provider_name} ({url}): {exc}")
        return {}


def _list_models(config: Any, provider_name: str, output_format: str) -> None:
    """List models from specified provider(s)."""
    if provider_name == "all":
        providers_to_fetch = list(config.providers.keys())
    elif provider_name not in config.providers:
        available = list(config.providers.keys())
        print(f"Error: Provider '{provider_name}' not found.")
        print(f"Available providers: {', '.join(available) if available else '(none)'}")
        raise SystemExit(1)
    else:
        providers_to_fetch = [provider_name]

    all_results: Dict[str, Any] = {}

    for pname in providers_to_fetch:
        pcfg = config.providers[pname]
        result = _fetch_provider_models(pname, pcfg)
        all_results[pname] = result

    if output_format == "json":
        print(json.dumps(all_results, indent=2))
    else:
        for pname, result in all_results.items():
            print(f"{pname}:")
            entries = result.get("data", [])
            if not isinstance(entries, list):
                print(f"  (no models found or invalid response)")
            elif len(entries) == 0:
                print(f"  (no models)")
            else:
                for entry in entries:
                    if isinstance(entry, dict):
                        model_id = entry.get("id", "?")
                        # Show additional fields if present
                        extra = {
                            k: v
                            for k, v in entry.items()
                            if k not in ("id", "object", "created", "owned_by")
                        }
                        if extra:
                            extra_str = ", ".join(f"{k}={v}" for k, v in extra.items())
                            print(f"  - {model_id} ({extra_str})")
                        else:
                            print(f"  - {model_id}")
            print()


def _app_factory():
    """Uvicorn factory for reload mode."""
    return create_app(
        os.getenv(_CONFIG_ENV_VAR, "config.yaml"),
        env_file=os.getenv(_ENV_FILE_ENV_VAR) or None,
    )


def main():
    """Main entry point for LazyRouter CLI."""
    parser = argparse.ArgumentParser(description="LazyRouter - Simplified LLM Router")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to (overrides config)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (overrides config)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on Python file changes (dev only)",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to environment file (default: auto-load .env if available)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models from provider(s) and exit (does not start server)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="all",
        help="Provider to list models from (default: all providers in config)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    args = parser.parse_args()

    config = load_config(args.config, env_file=args.env_file)

    if args.list_models:
        _list_models(config, args.provider, args.format)
        return

    # Determine host and port
    host = args.host or config.serve.host
    port = args.port or config.serve.port
    log_level = "debug" if config.serve.debug else "info"

    print(f"Starting LazyRouter server on {host}:{port}")
    print(f"Router model: {config.router.model}")
    print(f"Available models: {', '.join(config.llms.keys())}")
    if args.env_file:
        print(f"Environment file: {args.env_file}")
    print("\nEndpoints:")
    print(f"  - Health: http://{host}:{port}/health")
    print(f"  - Models: http://{host}:{port}/v1/models")
    print(f"  - Health Status: http://{host}:{port}/v1/health-status")
    print(f"  - Health Check: http://{host}:{port}/v1/health-check")
    print(f"  - Chat: http://{host}:{port}/v1/chat/completions")
    print(f"  - Anthropic: http://{host}:{port}/v1/messages")
    print(f"\nDocs: http://{host}:{port}/docs")

    # Run server
    if args.reload:
        os.environ[_CONFIG_ENV_VAR] = args.config
        if args.env_file:
            os.environ[_ENV_FILE_ENV_VAR] = args.env_file
        else:
            os.environ.pop(_ENV_FILE_ENV_VAR, None)
        print("\nAuto-reload: enabled")
        uvicorn.run(
            "lazyrouter.cli:_app_factory",
            host=host,
            port=port,
            reload=True,
            factory=True,
            log_level=log_level,
        )
    else:
        # Reuse the already-loaded config to avoid parsing YAML/dotenv twice.
        # env_file still matters because it was applied in load_config above.
        app = create_app(args.config, env_file=args.env_file, preloaded_config=config)
        uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
