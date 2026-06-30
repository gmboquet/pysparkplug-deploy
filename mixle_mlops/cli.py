"""Command-line entry points: run the gateway (`mixle-serve`) and admin tasks (`mixle-mlops ...`)."""
from __future__ import annotations

import argparse
import os

from .config import get_settings


def serve() -> None:
    import uvicorn

    get_settings()  # validate config early
    host = os.environ.get("MIXLE_HOST", "0.0.0.0")
    port = int(os.environ.get("MIXLE_PORT", "8000"))
    reload = os.environ.get("MIXLE_RELOAD", "0") == "1"
    uvicorn.run("mixle_mlops.gateway.app:app", host=host, port=port, reload=reload)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mixle-mlops", description="mixle-mlops platform admin")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="run the gateway")
    sub.add_parser("init-db", help="create database tables")
    cu = sub.add_parser("create-user", help="create a user and print an API key")
    cu.add_argument("email")
    cu.add_argument("password")
    cu.add_argument("--admin", action="store_true")
    ic = sub.add_parser("init-cloud", help="scaffold a provider-correct .env (object store + DB + redis)")
    ic.add_argument("provider", choices=["aws", "azure", "gcp", "alicloud", "local"])
    ic.add_argument("--dest", default=".env")
    ic.add_argument("--force", action="store_true", help="overwrite an existing .env")
    args = parser.parse_args(argv)

    if args.cmd == "serve":
        serve()
    elif args.cmd == "init-db":
        from .storage.db import init_db

        init_db()
        print("database initialized")
    elif args.cmd == "create-user":
        from sqlmodel import Session

        from .accounts import service
        from .storage.db import get_engine, init_db

        init_db()
        with Session(get_engine()) as session:
            user = service.create_user(session, args.email, args.password, is_admin=args.admin)
            _key, raw = service.create_api_key(session, user)
            print(f"user {user.email} created")
            print(f"api key: {raw}")
    elif args.cmd == "init-cloud":
        from .cloud_init import init_cloud, next_steps

        path = init_cloud(args.provider, dest=args.dest, overwrite=args.force)
        print(f"wrote {path}")
        print(next_steps(args.provider))
