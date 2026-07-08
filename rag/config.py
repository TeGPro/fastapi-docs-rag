from pathlib import Path

root_folder = Path.cwd()
DOCS_ROOT = root_folder / "data"
EXCLUDE_FILES = {"release-notes.md", "newsletter.md", "management-tasks.md"}