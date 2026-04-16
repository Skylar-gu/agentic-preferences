"""
check_job_status.py — Report whether 01_build.py output is ready.

Usage:  python check_job_status.py
Exit 0 if results exist, exit 1 if pending.
"""
import os, sys

RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results', 'pam_experiment.pkl')


def main():
    if os.path.exists(RESULTS_FILE):
        size_kb = os.path.getsize(RESULTS_FILE) / 1024
        print(f"DONE    {RESULTS_FILE}  ({size_kb:.1f} KB)")
        sys.exit(0)
    else:
        print(f"PENDING {RESULTS_FILE} not found — run 01_build.py first.")
        sys.exit(1)


if __name__ == '__main__':
    main()
