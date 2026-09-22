"""Installed JSON certification interface: python -m highs_turbo."""

import argparse
import json
from pathlib import Path
import sys

from highs_turbo.certification import certify, verify


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, json.dumps({"error": message}) + "\n")


def main(argv=None):
    parser = _Parser(description="Create and independently verify exact binary-model bounds")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("certify")
    create.add_argument("--input", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--seconds", type=float, default=2.0)
    create.add_argument("--seed", type=int, default=0)
    check = commands.add_parser("verify")
    check.add_argument("proof", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "certify":
            data = json.loads(args.input.read_text())
            if not isinstance(data, dict) or set(data) != {"model", "answer"}:
                raise ValueError("Input must contain exactly 'model' and 'answer'")
            bundle = certify(data["model"], data["answer"], time_limit=args.seconds, seed=args.seed)
            report = verify(bundle)
            args.output.write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n")
        else:
            report = verify(json.loads(args.proof.read_text()))
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, OverflowError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
