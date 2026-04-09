"""Package entrypoint for `python -m act_intent_policy`."""

from __future__ import annotations

import textwrap


def main() -> None:
    print(
        textwrap.dedent(
            """
            TurboPi ACT-Intent policy package

            Use one of:
              python -m act_intent_policy.train --help
              python -m act_intent_policy.eval --help
              python -m act_intent_policy.drive --help
            """
        ).strip()
    )


if __name__ == "__main__":
    main()
