import duckdb
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "governance.duckdb"

con = duckdb.connect(DB_PATH)

# --------------------------------------------------
# Helper: load CSV → DuckDB table
# --------------------------------------------------
def load_csv(table_name, csv_path):
    print(f"Loading {table_name}...")
    df = pd.read_csv(csv_path)
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")

# --------------------------------------------------
# Load base tables
# --------------------------------------------------
load_csv("dim_model", DATA_DIR / "dim_model.csv")
load_csv("dim_model_version", DATA_DIR / "dim_model_version.csv")
load_csv("dim_person", DATA_DIR / "dim_person.csv")
load_csv("bridge_model_assignment", DATA_DIR / "bridge_model_assignment.csv")
load_csv("fact_model_review", DATA_DIR / "fact_model_review.csv")

# --------------------------------------------------
# Load snapshot table
# --------------------------------------------------
load_csv(
    "fact_model_review_backlog_monthly",
    DATA_DIR / "fact_model_review_backlog_monthly.csv"
)

print("All datasets loaded into DuckDB successfully.")
con.close()
