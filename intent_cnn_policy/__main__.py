"""Package entrypoint for `python -m intent_cnn_policy`."""

from __future__ import annotations

import textwrap


def main() -> None:
    print(
        textwrap.dedent(
            """
            TurboPi intent-conditioned CNN policy package

            Use one of:
              python -m intent_cnn_policy.train --help
              python -m intent_cnn_policy.eval --help
              python -m intent_cnn_policy.drive --help
            """
        ).strip()
    )


if __name__ == "__main__":
    main()
