
import shutil
import os
from pathlib import Path
from dotenv import load_dotenv

import yaml
from roboflow import Roboflow


def build_class_mapping(source_classes, canonical_classes, class_map):
    """
    Create a mapping from source class IDs to canonical class IDs.

    Returns a dictionary like:
        {0: 1, 1: 0, 2: 5}
    """
    canonical_ids = {
        name: index
        for index, name in enumerate(canonical_classes)
    }

    mapping = {}

    for source_id, source_name in enumerate(source_classes):
        target_name = class_map.get(source_name)

        # Skip classes that aren't included
        if target_name is None:
            continue

        if target_name not in canonical_ids:
            raise ValueError(
                f"Unknown canonical class: {target_name}"
            )

        mapping[source_id] = canonical_ids[target_name]

    return mapping



def load_source_classes(data_yaml_path):
    """
    Read class names from an exported YOLO data.yaml file.

    Returns class names ordered by their numeric class IDs.
    """
    data_yaml_path = Path(data_yaml_path)

    with data_yaml_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    names = data["names"]

    # Some YOLO files use a list, others use a dictionary.
    if isinstance(names, list):
        return names

    if isinstance(names, dict):
        class_ids = [int(key) for key in names]

        if sorted(class_ids) != list(range(len(class_ids))):
            raise ValueError("Class IDs must be consecutive starting at 0")

        return [
            value
            for _, value in sorted(
                names.items(),
                key=lambda item: int(item[0])
            )
        ]

    raise ValueError("Unsupported class names format")



def convert_yolo_labels(input_path, output_path, mapping):
    """
    Convert a YOLO annotation file using a class-ID mapping.

    Labels without a matching class are excluded.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    converted_lines = []

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.strip().split()

            if not parts:
                continue

            if len(parts) != 5:
                raise ValueError(
                    f"{input_path}:{line_number}: "
                    "Expected 5 values per YOLO detection."
                )

            old_class_id = int(parts[0])

            # Drop classes not included in our mapping
            if old_class_id not in mapping:
                continue

            new_class_id = mapping[old_class_id]

            # Preserve the original bounding-box coordinates
            converted_lines.append(
                " ".join([str(new_class_id)] + parts[1:])
            )

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        if converted_lines:
            file.write("\n".join(converted_lines) + "\n")



def process_dataset(source_dir, output_dir, source_config, canonical_classes):
    """
    Copy source images into a standardized dataset and
    convert their annotation class IDs.

    Assuming a YOLO object-detection dataset.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # 1. Read the source dataset's class ordering
    source_classes = load_source_classes(source_dir / "data.yaml")

    # 2. Create the mapping from original IDs to our IDs
    mapping = build_class_mapping(
        source_classes,
        canonical_classes,
        source_config["class_map"]
    )

    # Roboflow often names its validation split "valid"
    split_names = {
        "train": "train",
        "valid": "val",
        "val": "val",
        "test": "test"
    }

    image_extensions = {".jpg", ".jpeg", ".png"}

    for source_split, output_split in split_names.items():
        images_dir = source_dir / source_split / "images"
        labels_dir = source_dir / source_split / "labels"

        if not images_dir.exists():
            continue

        # Only process splits approved in sources.yaml
        if output_split not in source_config["use"]:
            continue

        output_images = output_dir / "images" / output_split
        output_labels = output_dir / "labels" / output_split

        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)

        for image_path in images_dir.iterdir():
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in image_extensions:
                continue

            # Prefix names to avoid collisions across datasets
            new_stem = f"{source_config['name']}__{image_path.stem}"

            new_image = output_images / (
                new_stem + image_path.suffix
            )
            new_label = output_labels / (
                new_stem + ".txt"
            )

            original_label = labels_dir / (
                image_path.stem + ".txt"
            )

            if new_image.exists() or new_label.exists():
                raise FileExistsError(
                    f"Output already exists for {image_path.name}"
                )

            # Copy the original image
            shutil.copy2(image_path, new_image)

            if original_label.exists():
                # Convert its annotations
                convert_yolo_labels(
                    original_label,
                    new_label,
                    mapping
                )
            else:
                # An image without objects is a negative example
                new_label.write_text("", encoding="utf-8")

        print(f"Processed {source_config['name']}: {source_split}")



def main():
    load_dotenv()
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

        # Process the downloaded dataset
        process_dataset(
            source_dir=dataset.location,
            output_dir="datasets/merged",
            source_config=source,
            canonical_classes=config["canonical_classes"]
        )

        print(f"Finished processing {source['name']}")


if __name__ == "__main__":
    main()