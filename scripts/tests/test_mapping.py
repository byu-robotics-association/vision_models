from scripts.pull_dataset import build_class_mapping

source_classes = [
    "green_buoy",
    "red_buoy",
    "dock",
    "red_triangle"
]

canonical_classes = [
    "red_buoy",
    "green_buoy",
    "black_buoy",
    "yellow_buoy",
    "blue_buoy",
    "dock"
]

class_map = {
    "green_buoy": "green_buoy",
    "red_buoy": "red_buoy",
    "dock": "dock",
    "red_triangle": None
}

result = build_class_mapping(
    source_classes,
    canonical_classes,
    class_map
)

print(result)

assert result == {0: 1, 1: 0, 2: 5}
print("Class mapping test passed!")