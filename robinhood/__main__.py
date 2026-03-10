"""
Allow running as ``python -m robinhood``.

Routes to the unified CLI which provides subcommands::

    python -m robinhood collect  ...
    python -m robinhood train    ...
    python -m robinhood reformat ...
    python -m robinhood --version
"""
from robinhood.cli import main

main()
