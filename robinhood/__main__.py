"""
Allow running as:
    python -m robinhood              (pipeline)
    python -m robinhood train ...    (trainer directly)
"""
import sys

if len(sys.argv) > 1 and sys.argv[1] == "train":
    sys.argv = sys.argv[:1] + sys.argv[2:]  # strip "train" from argv
    from robinhood.trainer import main
    main()
else:
    from robinhood.pipeline import main
    main()
