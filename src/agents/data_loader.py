# data_loader.py (Modified - add sqlite elif)
import pandas as pd
from sqlalchemy import create_engine, text
from typing import Dict, Any, Optional
import os # Import os module

def get_db_connection_url(config: Dict[str, str]) -> Optional[str]:
    """Constructs a SQLAlchemy database URL from configuration."""
    db_type = config.get("type")
    if not db_type:
        return None

    # Basic URL structures - extend for more specific needs
    if db_type == "postgresql":
        url = f"postgresql://{config.get('user')}:{config.get('password')}@{config.get('host')}:{config.get('port', '5432')}/{config.get('database')}"
    elif db_type == "mysql":
         url = f"mysql+mysqlconnector://{config.get('user')}:{config.get('password')}@{config.get('host')}:{config.get('port', '3306')}/{config.get('database')}"
    elif db_type == "sqlserver":
        url = f"mssql+pyodbc://{config.get('user')}:{config.get('password')}@{config.get('host')}:{config.get('port', '1433')}/{config.get('database')}?driver=ODBC+Driver+17+for+SQL+Server" # Example
    # --- ADD THIS ELIF BLOCK ---
    elif db_type == "sqlite":
        db_path = config.get("database") # For SQLite, 'database' is the file path
        if not db_path:
             return None # Path is required

        # Ensure the path is absolute if it exists, or handle relative paths carefully
        # This example assumes 'database' in config is the file path
        # You might want to use os.path.abspath(db_path) if db_path could be relative
        # For simplicity here, we'll use the path as provided:
        url = f"sqlite:///{db_path}"
    # --- END ADDITION ---
    else:
        return None # Unsupported database type

    return url

# load_data_from_db function remains the same
def load_data_from_db(db_config: Dict[str, str], query: str) -> pd.DataFrame:
    """
    Loads data from a database using the provided configuration and query.

    Args:
        db_config: Dictionary containing database connection details (type, host, port, database, user, password).
                   For SQLite, only 'type' and 'database' (file path) are needed.
        query: The SQL query to execute or table name to read.

    Returns:
        A pandas DataFrame containing the query results.

    Raises:
        ValueError: If database type is unsupported or config is incomplete.
        Exception: For any database connection or query execution errors.
    """
    db_url = get_db_connection_url(db_config)

    if not db_url:
        # Refine error message for SQLite if path is missing
        if db_config.get("type") == "sqlite" and not db_config.get("database"):
             raise ValueError("Database file path ('database' field) is required for SQLite.")
        else:
            raise ValueError(f"Unsupported database type or incomplete config: {db_config.get('type')}")

    engine = None
    try:
        # Create engine
        # For SQLite, check_same_thread=False is often needed for Streamlit
        engine = create_engine(db_url, connect_args={"check_same_thread": False} if db_config.get("type") == "sqlite" else {})

        # Use text() for the query for better compatibility and safety
        with engine.connect() as connection:
            # For SQLite, if the query is just a table name, we need to handle it differently
            if db_config.get("type") == "sqlite" and not query.strip().upper().startswith(("SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER")):
                # If it's a table name, first verify it exists
                table_name = query.strip()
                check_table_query = text(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                result = connection.execute(check_table_query).fetchone()
                
                if not result:
                    raise ValueError(f"Table '{table_name}' does not exist in the database")
                
                # If table exists, select all from it
                df = pd.read_sql_table(table_name, connection)
            else:
                # For regular queries or non-SQLite databases
                df = pd.read_sql(text(query), connection)

        return df

    except Exception as e:
        print(f"Database loading error: {e}")
        raise Exception(f"Failed to load data from database: {e}")
    finally:
        # Dispose the engine connection pool
        if engine:
            engine.dispose()

# Example Usage (for testing data_loader.py standalone) - Update path for SQLite
if __name__ == '__main__':
    # Replace with your test database configuration
    test_db_config = {
        "type": "sqlite",
        "database": "my_local_database.db" # Replace with path to your .db file
    }
    test_query = "SELECT * FROM mytable LIMIT 10" # or a table name "mytable"

    try:
        print(f"Attempting to load data using query: {test_query}")
        df = load_data_from_db(test_db_config, test_query)
        print("Data loaded successfully:")
        print(df.head())
    except Exception as e:
        print(f"Failed to load data: {e}")