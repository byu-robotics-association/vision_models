
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.pull_dataset import load_source_classes


with TemporaryDirectory() as temp_dir:
    yaml_file = Path(temp_dir) / "data.yaml"

    yaml_file.write_text(
        """
names:
  2: dock
  0: red_buoy
  1: green_buoy
""",
        encoding="utf-8"
    )

    result = load_source_classes(yaml_file)

    expected = [
        "red_buoy",
        "green_buoy",
        "dock"
    ]

    assert result == expected, (
        f"Expected {expected}, got {result}"
    )

    print("Source class loading test passed!")
    print(result)
