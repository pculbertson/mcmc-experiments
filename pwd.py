from pathlib import Path
import pathspec

def load_gitignore(gitignore_path=".gitignore"):
    """Load the .gitignore rules if the file exists."""
    try:
        with open(gitignore_path, "r") as f:
            gitignore_content = f.read()
        return pathspec.PathSpec.from_lines("gitwildmatch", gitignore_content.splitlines())
    except FileNotFoundError:
        return pathspec.PathSpec.from_lines("gitwildmatch", [])  # No rules if .gitignore missing

def pretty_print_dir(path, spec, root, indent=0):
    """Recursively print directory contents, excluding matches from .gitignore and `.git/`."""
    for item in path.iterdir():
        # Skip the `.git` directory explicitly
        if item.name == ".git":
            continue

        # Compute the relative path from the project root for matching with .gitignore rules
        rel_path = item.relative_to(root)

        if spec.match_file(str(rel_path)):
            continue  # Skip items matched by .gitignore

        print(" " * indent + f"|- {item.name}")
        if item.is_dir():
            pretty_print_dir(item, spec, root, indent + 4)

if __name__ == "__main__":
    root = Path(".").resolve()  # Ensure we work with absolute paths
    gitignore_spec = load_gitignore(root / ".gitignore")  # Load .gitignore

    print("Current Directory Structure (excluding .git and .gitignore contents):")
    pretty_print_dir(root, gitignore_spec, root)
