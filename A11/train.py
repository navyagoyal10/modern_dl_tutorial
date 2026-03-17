"""Training entrypoint for A11."""

from __future__ import annotations

from finetune import main, parse_args


if __name__ == "__main__":
    main(parse_args())
