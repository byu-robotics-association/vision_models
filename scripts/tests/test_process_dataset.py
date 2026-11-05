
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.pull_dataset import process_dataset


def test_process_dataset():
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source = root / "source"
        output = root / "merged"

        # Build a tiny fake dataset
        (source / "train" / "images").mkdir(parents=True)
        (source / "train" / "labels").mkdir(parents=True)

        (source / "data.yaml").write_text(
            "names: [green_buoy, red_buoy, dock]\n",
            encoding="utf-8"
        )

        # The image contents are irrelevant to the copy test
        (source / "train" / "images" / "example.jpg").write_bytes(
            b"fake image data"
        )

        (source / "train" / "labels" / "example.txt").write_text(
            "0 0.50 0.50 0.20 0.20\n"
            "1 0.30 0.40 0.10 0.10\n",
            encoding="utf-8"
        )

        source_config = {
            "name": "test-source",
            "use": ["train"],
            "class_map": {
                "green_buoy": "green_buoy",
                "red_buoy": "red_buoy",
                "dock": "dock"
            }
        }

        canonical_classes = [
            "red_buoy",
            "green_buoy",
            "dock"
        ]

        process_dataset(
            source,
            output,
            source_config,
            canonical_classes
        )

        # Verify image was copied
        new_image = (
            output / "images" / "train" / "test-source__example.jpg"
        )
        assert new_image.read_bytes() == b"fake image data"

        # Verify annotation class IDs were converted
        new_label = (
            output / "labels" / "train" / "test-source__example.txt"
        )

        assert new_label.read_text(encoding="utf-8") == (
            "1 0.50 0.50 0.20 0.20\n"
            "0 0.30 0.40 0.10 0.10\n"
        )
