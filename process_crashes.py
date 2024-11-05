import os
import logging
import requests
import zipfile
import duckdb
import pandas as pd
import argparse
from pathlib import Path


def setup_directories():
    """
    Create remaining directories in the structure, respecting existing ones.
    Returns the root path and dictionary of paths.
    """
    root_path = Path.cwd()
    
    # Define directory structure
    dirs = {
        'data': {  # Skip if exists
            'downloads': None,
            'extracted': None
        },
        'output': None,
        'logs': None
    }
    
    paths = {}
    
    # Create only missing directories
    logging.info("Setting up remaining directories...")
    for main_dir, subdirs in dirs.items():
        main_path = root_path / main_dir
        
        # Skip directory creation if it exists
        if not main_path.exists():
            main_path.mkdir(exist_ok=True)
            logging.info(f"Created {main_dir}/")
        
        paths[main_dir] = main_path
        
        # Create subdirectories if any
        if subdirs:
            for subdir in subdirs:
                sub_path = main_path / subdir
                if not sub_path.exists():
                    sub_path.mkdir(exist_ok=True)
                    logging.info(f"Created {main_dir}/{subdir}/")
                paths[f'{main_dir}/{subdir}'] = sub_path
    
    logging.info("Directory setup complete!")
    return root_path, paths


def setup_logging():
    """Configure logging to both file and console"""
    # File logging - detailed
    file_handler = logging.FileHandler('./logs/processing.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Console logging - progress only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Root logger setup
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def download_data(years, counties, table_names, base_url):
    """Download and extract NJDOT crash data tables"""
    total_downloads = len(years) * len(counties) * len(table_names)
    current = 0
    
    for year in years:
        for county in counties:
            for table in table_names:
                current += 1
                logging.info(f"Downloading {county} {year} {table} ({current}/{total_downloads})")
                
                url = base_url.format(year=year, county=county, table=table)
                zip_path = Path('data', 'downloads', f'{county}_{year}_{table}.zip')
                extract_path = Path('data', 'extracted', f'{county}_{year}_{table}')
                
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    
                    zip_path.write_bytes(response.content)
                    extract_path.mkdir(exist_ok=True)
                    
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_path)
                    
                except Exception as e:
                    logging.error(f'Failed to process {url}: {str(e)}')


class DataProcessor:
    def __init__(self):
        self.conn = None
    

    def load_field_positions(self, csv_path):
        """Load field fixed width values"""
        field_positions = []
        field_names = []
        with open(csv_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                field_names.append(parts[0].strip())
                start = int(parts[1].strip()) - 1
                end = int(parts[2].strip())
                field_positions.append((start, end))
        return field_names, field_positions


    def parse_line(self, line, colspecs, field_names):
        """Parse lines in the txt files"""
        return {
            name: line[start:end].strip() if start < len(line) else ''
            for (start, end), name in zip(colspecs, field_names)
        }


    def review_file(self, file_path, field_names_path):
        """Review the txt file and handle line length issues
     
        1. Handles multiline records by joining broken lines
        2. Validates and repairs line lengths
        3. Keeps detailed logs of modifications
        4. Preserves more data while maintaining data quality
        """
        field_names, colspecs = self.load_field_positions(field_names_path)
        expected_length = max(end for _, end in colspecs)
        
        data = []
        issues = []
        buffer = ""
        
        with open(file_path, 'r', encoding='latin1') as f:
            for line_num, line in enumerate(f, 1):
                line = line.rstrip('\n\r')
                
                # Check if this line might be a continuation of previous line
                if buffer:
                    # If line starts with a space or known continuation character
                    if line.startswith(' ') or not any(line.startswith(str(i)) for i in range(10)):
                        buffer += line
                        issues.append(f"Line {line_num} merged with previous line")
                        continue
                    else:
                        # Process the buffered line before starting new record
                        if len(buffer) > 0:
                            processed_record = self._process_record(buffer, expected_length, line_num-1, 
                                                                colspecs, field_names, issues)
                            if processed_record:
                                data.append(processed_record)
                        buffer = ""
                
                # Start new record
                if len(line) < expected_length:
                    buffer = line  # Maybe incomplete - store in buffer
                    continue
                elif len(line) > expected_length:
                    # Line is too long - might contain embedded newline
                    buffer = line[:expected_length]
                    issues.append(f"Line {line_num} truncated from {len(line)} to {expected_length} characters")
                else:
                    buffer = line
                
                # Process complete record
                if len(buffer) == expected_length:
                    processed_record = self._process_record(buffer, expected_length, line_num,
                                                        colspecs, field_names, issues)
                    if processed_record:
                        data.append(processed_record)
                    buffer = ""
            
            # Process any remaining buffer at end of file
            if buffer:
                processed_record = self._process_record(buffer, expected_length, line_num,
                                                    colspecs, field_names, issues)
                if processed_record:
                    data.append(processed_record)
        
        return data, issues
    
    def _process_record(self, line, expected_length, line_num, colspecs, field_names, issues):
        """Process a single record line, handling any necessary repairs
        
        Returns:
        - dict: Processed record if valid
        - None: If record should be skipped
        """
        # Validate basic record structure
        if not line.strip():
            issues.append(f"Line {line_num}: Empty line skipped")
            return None
            
        # Handle length issues
        if len(line) < expected_length:
            line = line.ljust(expected_length)
            issues.append(f"Line {line_num}: Padded short line from {len(line)} to {expected_length} characters")
        elif len(line) > expected_length:
            line = line[:expected_length]
            issues.append(f"Line {line_num}: Truncated long line from {len(line)} to {expected_length} characters")
        
        # Parse fields
        record = {}
        for (start, end), name in zip(colspecs, field_names):
            value = line[start:end].strip() if start < len(line) else ''
            
            # Additional validation could be added here based on field type
            record[name] = value
        
        # Validate critical fields (customize based on your needs)
        if 'casenumber' in record and record['casenumber'].startswith(','):
            issues.append(f"Line {line_num}: Invalid case number format")
            return None
            
        if 'police_dept_code' in record and not record['police_dept_code'].strip():
            issues.append(f"Line {line_num}: Missing police department code")
            return None
        
        return record


    def process_txts(self, years, counties, table_names):
        """Process each txt year/county/table"""
        combined_data = {table: [] for table in table_names}
        all_issues = {table: [] for table in table_names}
        
        total_files = len(years) * len(counties) * len(table_names)
        current = 0
        
        for year in years:
            for county in counties:
                for table in table_names:
                    current += 1
                    logging.info(f"Processing {county} {year} {table} ({current}/{total_files})")
                    
                    file_path = Path('data', 'extracted', f'{county}_{year}_{table}', f'{county}{year}{table}.txt')
                    field_names_path = Path('data', 'fields', f'{table}.csv')
                    
                    if not all(p.exists() for p in [file_path, field_names_path]):
                        logging.warning(f"Missing files for {county} {year} {table}")
                        continue
                    
                    data, issues = self.review_file(file_path, field_names_path)
                    combined_data[table].extend(data)
                    all_issues[table].extend([f"{county}_{year}: {issue}" for issue in issues])

        # Write issues to separate log files
        for table, issues in all_issues.items():
            issue_log_path = Path('logs', f'{table}_import_issues.log')
            issue_log_path.write_text('\n'.join(issues))

        return combined_data


    def init_database(self):
        """Initialize DuckDB database"""
        self.conn = duckdb.connect('data/crashes.duckdb')
        self.conn.install_extension('spatial')
        self.conn.load_extension('spatial')


    def load_data_to_db(self, combined_data):
        """Load processed data into DuckDB"""
        total_tables = len(combined_data)
        current = 0
        
        for table, data in combined_data.items():
            current += 1
            logging.info(f"Loading {table} to database ({current}/{total_tables})")
            
            df = pd.DataFrame(data)
            table_name = f'crash_nj_{table.lower()}'
            
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
            
            count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            logging.info(f"Loaded {count} records into {table_name}")
    

    def load_nj_roads_to_duckdb(self):
        """Download NJ roads shapefile and load into DuckDB directly"""
        logging.info("Downloading NJDOT Linear Referencing System")
        url = "https://www.nj.gov/transportation/refdata/gis/zip/NJ_Roads_shp.zip"
        response = requests.get(url)
        
        if response.status_code == 200:
            # Save and extract zip file
            with open("data/downloads/NJ_Roads_shp.zip", 'wb') as f:
                f.write(response.content)
            
            with zipfile.ZipFile("data/downloads/NJ_Roads_shp.zip", 'r') as zip_ref:
                zip_ref.extractall('data/extracted')
            
            logging.info("Loading roads to DuckDB...")
            self.conn.install_extension('spatial')
            self.conn.load_extension('spatial')

            self.conn.execute("DROP TABLE IF EXISTS nj_lrs")
            
            self.conn.execute("""
                CREATE TABLE nj_lrs AS 
                SELECT *
                FROM ST_Read('data/extracted/NJ_Roads shp/NJ_Roads_shp.shp');
            """)

            count = self.conn.execute("SELECT COUNT(*) FROM nj_lrs").fetchone()[0]
            logging.info(f"Loaded {count} records into nj_lrs")


    def process_geometries(self):
        """Process crash geometries"""
        logging.info("Processing geometries from Linear Referencing System")
        
        self.conn.execute("ALTER TABLE crash_nj_accidents ADD COLUMN IF NOT EXISTS geom GEOMETRY")
        
        counties = self.conn.execute("""
            SELECT DISTINCT SUBSTRING(sri_std_rte_identifier, 1, 2) as county_code
            FROM crash_nj_accidents
            WHERE length(sri_std_rte_identifier) >= 2
            ORDER BY county_code;
        """).fetchall()
        
        total_counties = len(counties)
        for idx, county in enumerate(counties, 1):
            self.process_county_crashes(county[0])


    def process_county_crashes(self, county_code):
        """Process crashes for a single county"""
        total_crashes = self.conn.execute("""
            SELECT COUNT(*)
            FROM crash_nj_accidents
            WHERE SUBSTRING(sri_std_rte_identifier, 1, 2) = ?
        """, [county_code]).fetchone()[0]
        
        batch_size = 500
        for offset in range(0, total_crashes, batch_size):
            self.process_crash_batch(county_code, batch_size, offset)


    def process_crash_batch(self, county_code, batch_size, offset):
        """Process a batch of crashes for spatial analysis"""
        try:
            self.conn.execute("""
                INSTALL spatial;
                LOAD spatial;
                WITH road_vertices AS (
                    -- Get vertices only for roads in current county
                    SELECT 
                        sri,
                        unnest(generate_series(1, ST_NPoints(geom))) as point_number,
                        ST_PointN(geom, unnest(generate_series(1, ST_NPoints(geom)))::int) as vertex,
                        ST_M(ST_PointN(geom, unnest(generate_series(1, ST_NPoints(geom)))::int)) as m_value
                    FROM nj_lrs
                    WHERE SUBSTRING(sri, 1, 2) = ?
                ),
                batch_crashes AS (
                    -- Get just the crashes for this batch
                    SELECT *
                    FROM crash_nj_accidents
                    WHERE SUBSTRING(sri_std_rte_identifier, 1, 2) = ?
                    ORDER BY casenumber
                    LIMIT ? OFFSET ?
                ),
                nearest_vertices AS (
                    SELECT 
                        a.casenumber,
                        a.sri_std_rte_identifier as sri,
                        a.milepost::numeric as milepost,
                        FIRST_VALUE(rv.vertex) OVER (
                            PARTITION BY a.casenumber
                            ORDER BY 
                                CASE 
                                    WHEN rv.m_value <= a.milepost::numeric THEN rv.m_value 
                                    ELSE -999999 
                                END DESC
                        ) as point_before,
                        FIRST_VALUE(rv.vertex) OVER (
                            PARTITION BY a.casenumber
                            ORDER BY 
                                CASE 
                                    WHEN rv.m_value >= a.milepost::numeric THEN rv.m_value
                                    ELSE 999999 
                                END ASC
                        ) as point_after,
                        FIRST_VALUE(rv.m_value) OVER (
                            PARTITION BY a.casenumber
                            ORDER BY 
                                CASE 
                                    WHEN rv.m_value <= a.milepost::numeric THEN rv.m_value 
                                    ELSE -999999 
                                END DESC
                        ) as m_before,
                        FIRST_VALUE(rv.m_value) OVER (
                            PARTITION BY a.casenumber
                            ORDER BY 
                                CASE 
                                    WHEN rv.m_value >= a.milepost::numeric THEN rv.m_value
                                    ELSE 999999
                                END ASC
                        ) as m_after
                    FROM batch_crashes a
                    JOIN road_vertices rv ON a.sri_std_rte_identifier = rv.sri
                    WHERE length(a.sri_std_rte_identifier) != 0
                    AND length(a.milepost) != 0
                )
                UPDATE crash_nj_accidents a
                SET geom = 
                    CASE 
                        WHEN nv.m_before = nv.milepost THEN nv.point_before
                        WHEN nv.m_after = nv.milepost THEN nv.point_after
                        ELSE ST_Point(
                            ST_X(nv.point_before) + (ST_X(nv.point_after) - ST_X(nv.point_before)) * 
                            (nv.milepost - nv.m_before) / NULLIF((nv.m_after - nv.m_before), 0),
                            ST_Y(nv.point_before) + (ST_Y(nv.point_after) - ST_Y(nv.point_before)) * 
                            (nv.milepost - nv.m_before) / NULLIF((nv.m_after - nv.m_before), 0)
                        )
                    END
                FROM nearest_vertices nv
                WHERE a.casenumber = nv.casenumber;
            """, [county_code, county_code, batch_size, offset])
                
        except Exception as e:
            raise


    def export_data(self):
        """Export processed data"""
        logging.debug("Exporting geojson and csv to output folder")
            # Export accidents as CSV with wkt geometry
        self.conn.execute("""
                COPY (
                    SELECT 
                        * EXCLUDE (geom),
                        ST_Force2D(ST_Transform(geom, 'EPSG:3424', 'EPSG:4326', always_xy := true)) as geometry
                    FROM crash_nj_accidents
                ) TO 'output/crash_nj_accidents.geojson'
                WITH (FORMAT GDAL, DRIVER 'GeoJSON', SRS 'EPSG:4326');
            """)
        
        # Export other tables as CSV
        other_tables = ['accidents','drivers', 'pedestrians', 'occupants', 'vehicles']
        for table in other_tables:
            self.conn.execute(f"""
                COPY crash_nj_{table} 
                TO 'output/crash_nj_{table}.csv'
                WITH (FORMAT CSV, HEADER);
                """)


    def cleanup(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process NJ DOT crash data')
    parser.add_argument('--start-year', type=int, default=2017,
                      help='Start year for data processing (default: 2017)')
    parser.add_argument('--end-year', type=int, default=2022,
                      help='End year for data processing (default: 2022)')
    parser.add_argument('--counties', type=str, 
                      default='Burlington,Camden,Gloucester,Mercer',
                      help='Comma-separated list of counties (default: Burlington,Camden,Gloucester,Mercer)')
    return parser.parse_args()


def main():
    """Process NJ DOT crash data"""
    try:
        # Parse arguments
        args = parse_args()
        
        # Setup
        setup_directories()
        setup_logging()
        
        # # Configuration
        years = range(args.start_year, args.end_year + 1)
        counties_list = [c.strip() for c in args.counties.split(',')]
        table_names = ['Accidents', 'Drivers', 'Pedestrians', 'Occupants', 'Vehicles']
        base_url = 'https://www.state.nj.us/transportation/refdata/accident/{year}/{county}{year}{table}.zip'
        
        # # Processing
        logging.info("Starting NJ Crash Data Processing")
        
        # download_data(years, counties_list, table_names, base_url)
        
        processor = DataProcessor()
        # combined_data = processor.process_txts(years, counties_list, table_names)
        
        processor.init_database()
        # processor.load_data_to_db(combined_data)
        # processor.load_nj_roads_to_duckdb()
        # processor.process_geometries()
        processor.export_data()
        processor.cleanup()
        
        logging.info("Processing Complete")
        
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}", exc_info=True)
        logging.info("Processing failed. Check logs for details.")
        raise

if __name__ == "__main__":
    main()