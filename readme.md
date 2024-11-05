# DVRPC NJ Crash Data Processor
A Python script for downloading and processing New Jersey Department of Transportation (NJDOT) crash data. This tool downloads crash data by county and year, processes the fixed-width text files, and loads them into a DuckDB database to output csv and geojson files.

## Features

1. Downloads crash data from NJDOT's public website

2. Processes data tables:

    - Accidents
    - Drivers
    - Pedestrians
    - Occupants
    - Vehicles

3. Integrates with the NJDOT Linear Referencing System (LRS) for generating geometry (requires SRI and milepost)
4. Exports processed data in GeoJSON and CSV formats
5. Detailed logging system for tracking processing status and issues

## Run

1. Clone the repo
    ``` 
    git clone https://github.com/dvrpc/nj-crash-data.git
    ```
2. Create a Python virtual environment with dependencies

    Working in the repo directory from your terminal:

   ```
   cd \nj-crash-data
   ```
    - create new venv
    ```
    python -m venv venv
    ```
    - activate venv
    ```
    .\venv\scripts\activate
    ```
    - install requirements
    ```
    pip install -r requirements.txt
    ```
3. Basic usage with default settings:
    ```
    python process_crashes.py
    ```
    Customize the year range and counties:
    ```
    python process_crashes.py --start-year 2018 --end-year 2022 --counties "Burlington,Camden"
    ```

## Command Line Arguments

`--start-year`: Start year of data (default: 2017)

`--end-year`: End year of data (default: 2022)

`--counties`: Comma-separated list of counties (default: "Burlington,Camden,Gloucester,Mercer")

## Directory Structure
The script creates and uses the following directory structure:

```
├── data/
│   ├── downloads/     # Downloaded zip files
│   ├── extracted/     # Extracted files
│   └── fields/        # Field definition files
├── output/           # Processed data outputs
└── logs/            # Processing logs
```

## Output Files

- output/accidents.geojson: Spatial data for all crashes able to be mapped with SRI/milepost
- output/accidents.csv: Crash table
- output/drivers.csv: Driver table
- output/pedestrians.csv: Pedestrian table
- output/occupants.csv: Occupant table
- output/vehicles.csv: Vehicle table

## Logging

General processing logs: `logs/processing.log`

Individual table import issues: `logs/{table}_import_issues.log`

## Data Source
This tool uses publicly available crash data from the New Jersey Department of Transportation:

- Crash Data: https://www.state.nj.us/transportation/refdata/accident/
- Linear Referencing Data: https://www.nj.gov/transportation/refdata/gis/data.shtm

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Notes
The spatial processing relies on NJDOT's Linear Referencing System

Processing large date ranges or multiple counties can be time intensive

All data is processed locally; no external databases required

## Acknowledgments
This tool is built upon the public data provided by the New Jersey Department of Transportation.
