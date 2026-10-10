import json
import sys
from typing import Any

import jsonschema
from jsonschema import validate


def validate_manifest(manifest: Any, schema: Any) -> bool:
    try:
        validate(instance=manifest, schema=schema)
        return True
    except jsonschema.ValidationError as e:
        print(f'Invalid manifest.json: {e.message}')
        return False
    except jsonschema.SchemaError as e:
        print(f'Invalid Schema: {e.message}')
        return False

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python validate_manifest.py <manifest_path> <schema_path>')
        sys.exit(1)

    manifest_path = sys.argv[1]
    schema_path = sys.argv[2]

    try:
        with open(manifest_path) as manifest_file, open(schema_path) as schema_file:
            manifest = json.load(manifest_file)
            schema = json.load(schema_file)
            if validate_manifest(manifest, schema):
                print(f"'{manifest_path.split('/')[-1]}' validated successfully.")
                sys.exit(0)
            sys.exit(1)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
