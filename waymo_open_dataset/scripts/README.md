# Waymo Open Dataset Download Script
This script automates the download of individual segments of data from google cloud storage instead of downloading it as one chunk.

## Prerequisite
* Python 3
* gsutil Follow [this](https://cloud.google.com/sdk/docs/quickstart) tutorial to install google cloud sdk
* access to Waymo Open Dataset (follow the [Quick Start](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md))

## How to use
* Before you run the following code, make sure you have set up gsutil and waymo_open_dataset!
* choose which type of segments you need to download [training, validation, testing]

example:
`python access_data_using_gsutil.py training --out-dir /dataset/training`
