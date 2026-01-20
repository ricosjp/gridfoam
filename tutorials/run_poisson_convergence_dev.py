import os
import pathlib
import subprocess
import sys

from jinja2 import Environment, FileSystemLoader


def main() -> None:
    """Run poisson.py for depth_limit from 0 to 5."""
    # Setup paths
    script_dir = pathlib.Path(__file__).parent
    project_root = script_dir.parent
    template_dir = project_root / "tests" / "data" / "yaml" / "SpatialConvergenceTest"
    # template_file = "poisson.yaml.j2"
    template_file = "adaptive_poisson.yaml.j2"
    poisson_script = script_dir / "poisson.py"

    # Change to project root directory
    original_cwd = pathlib.Path.cwd()
    os.chdir(project_root)

    try:
        # Setup jinja2 environment
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        template = env.get_template(template_file)

        # Run for each depth_limit
        for depth_limit in range(6):  # 0 to 5
            print(f"\n{'='*60}")
            print(f"Running with depth_limit = {depth_limit}")
            print(f"{'='*60}")

            # Render template with current depth_limit
            rendered_yaml = template.render(depth_limit=depth_limit)

            # Write to temporary file
            temp_yaml = template_dir / "poisson.yaml"
            temp_yaml.write_text(rendered_yaml)

            try:
                # Run poisson.py sequentially (subprocess.run blocks until completion)
                result = subprocess.run(
                    [
                        sys.executable,
                        str(poisson_script),
                    ],
                    cwd=str(project_root),
                    check=False,
                )

                # Flush output after process completes to ensure sequential execution
                sys.stdout.flush()
                sys.stderr.flush()

                if result.returncode != 0:
                    print(
                        f"Warning: poisson.py failed with depth_limit={depth_limit}",
                        file=sys.stderr,
                    )
                else:
                    print(f"Successfully completed depth_limit={depth_limit}")

            finally:
                # Clean up temporary file (optional - comment out if you want to keep them)
                # temp_yaml.unlink()
                pass

    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
