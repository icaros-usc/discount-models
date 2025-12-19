"""Makes all warnings show tracebacks.

https://gist.github.com/planetceres/72c0f721292762e019236ed631ad0182

Usage:
    import src.utils.warn_traceback  # pylint: disable = unused-import
"""

import sys
import traceback
import warnings


def warn_with_traceback(message,
                        category,
                        filename,
                        lineno,
                        file=None,
                        line=None):

    # We use stdout here so that the output shows up in tune with logging
    # statements. However, warnings should really go to sys.stderr.
    log = file if hasattr(file, 'write') else sys.stdout

    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


warnings.showwarning = warn_with_traceback
