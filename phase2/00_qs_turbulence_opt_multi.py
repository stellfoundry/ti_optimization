"""Compatibility wrapper for the renamed phase2 optimization driver."""

from qs_turbulence_opt_multi import main


if __name__ == "__main__":
    raise SystemExit(main())
