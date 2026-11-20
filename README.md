# Vision Models

Training and testing of YOLO models for `y_boat_core`.

## Setup

Run `uv sync` to install the necessary dependencies.  You may need to run `brew install uv`, or a windows equivalent to instal it first.

### RoboFlow

In order to download datasets, you'll need an API secret from RoboFlow.  Create your `.env` file, then put your own API secret in that file.  **Do not put your secret in the `example.env` or commit it**.

```bash
cp example.env .env
```