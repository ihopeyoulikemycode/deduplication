import os

from dotenv import load_dotenv
from pyspark.sql import SparkSession
from sqlalchemy import create_engine, text

from constants import FILE_PATH

load_dotenv()

# Initialize Spark session
spark = (
    SparkSession.builder.appName("ParquetToPostgres").master("local[*]").getOrCreate()
)

# Read Parquet data into a Spark DataFrame
df_parquet = spark.read.parquet(FILE_PATH)
df = df_parquet.toPandas()

# Set up PostgreSQL connection with SQLAlchemy
pg_connection_string = f"postgresql://{os.environ.get('PG_USER')}:{os.environ.get('PG_PASS')}@localhost:5432/prod"

# Create a connection engine to PostgreSQL
engine = create_engine(pg_connection_string)

# Ensure schema exists before inserting
schema_name = "prod_data"
with engine.connect() as conn:
    conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name};"))

# Load the DataFrame into PostgreSQL
df.to_sql(
    name="init_prod_deduped",
    schema=schema_name,
    con=engine,
    index=False,
    if_exists="append",  # Append data to the existing table
    method="multi",
)

# 7. Confirm success
print("Data successfully loaded into PostgreSQL")
