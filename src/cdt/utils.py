"""Utility functions for Commercial Debt Tracker (CDT) project."""

import subprocess


def get_files_last_commit_hash(file_path: str) -> str:
    """Returns the commit hash of the last commit that modified the given file.

    Parameters:
      file_path: Path to the file relative to the Git repository root.

    Returns:
      str: The commit hash as a string, or None if an error occurs.
    """
    try:
        # Run the git log command to get the last commit hash for the file.
        commit_hash = subprocess.check_output(  # noqa: S603
            ["git", "log", "-1", "--pretty=format:%H", file_path],  # noqa: S607
            stderr=subprocess.STDOUT,
        )
        return commit_hash.decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        print(f"Error retrieving commit hash for {file_path}: {e.output.decode()}")
        return None
