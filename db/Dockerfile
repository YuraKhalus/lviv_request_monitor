# Use the lightweight postgres:15-alpine image
FROM postgres:15-alpine

# Copy the initialization script and the cleaned data into the container.
# Scripts in /docker-entrypoint-initdb.d are automatically executed on first run.
COPY ./db/init.sql /docker-entrypoint-initdb.d/
COPY ./data/cleaned_appeals.csv /docker-entrypoint-initdb.d/
