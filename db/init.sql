-- Drop the table if it exists to ensure a clean slate on re-initialization
DROP TABLE IF EXISTS appeals;

-- Create the table structure for the appeals data
CREATE TABLE appeals (
    registrationDate TIMESTAMP,
    executionDate TIMESTAMP,
    district VARCHAR(100),
    category VARCHAR(255), -- Using 255 for safety with long category names
    days_to_resolve FLOAT,
    latitude FLOAT,
    longitude FLOAT
);

-- Copy data from the CSV file into the 'appeals' table.
-- The file path '/docker-entrypoint-initdb.d/cleaned_appeals.csv' is relative to the container's filesystem.
COPY appeals(registrationDate, executionDate, district, category, days_to_resolve, latitude, longitude)
FROM '/docker-entrypoint-initdb.d/cleaned_appeals.csv'
DELIMITER ','
CSV HEADER;
