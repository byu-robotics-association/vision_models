
import os
from pathlib import Path

import yaml
from roboflow import Roboflow


def main():
    # Read dataset configuration
    config_path = Path("datasets/sources.yaml")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Read API key from environment
    api_key = os.environ.get("ROBOFLOW_API_KEY")

    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY environment variable is missing."
        )

    rf = Roboflow(api_key=api_key)

    # Process each configured source
    for source in config["sources"]:
        print(f"Processing: {source['name']}")

        workspace = source["workspace"]
        project_name = source["project"]
        version_number = source["version"]
        export_format = source["format"]

        project = rf.workspace(workspace).project(project_name)
        version = project.version(version_number)

        # Download the specified dataset
        dataset = version.download(
            export_format,
            location=f"datasets/raw/{source['name']}"
        )

        print(f"Downloaded to: {dataset.location}")


if __name__ == "__main__":
    main()