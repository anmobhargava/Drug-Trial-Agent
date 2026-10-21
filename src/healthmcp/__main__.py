"""Run the server:  python -m healthmcp [--demo-data] [--http]

  default      real openFDA + ClinicalTrials.gov data, over stdio (what Claude Desktop and agents use)
  --demo-data  fictional data, no network needed
  --http       serve over streamable HTTP on port 8000 instead of stdio
"""

import argparse

from .api import LiveAPI
from .demo_data import DemoAPI
from .server import build_server


def main():
    ap = argparse.ArgumentParser(prog="healthmcp")
    ap.add_argument("--demo-data", action="store_true")
    ap.add_argument("--http", action="store_true")
    a = ap.parse_args()
    server = build_server(DemoAPI() if a.demo_data else LiveAPI())
    server.run(transport="streamable-http" if a.http else "stdio")


if __name__ == "__main__":
    main()
