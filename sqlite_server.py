import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

mcp = FastMCP("sqlite")
DB_PATH = "test.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@mcp.tool()
def write_query(query: str, parameters: Optional[List[Any]] = None) -> str:
    """
    Execute a write SQL query (INSERT, UPDATE, DELETE, CREATE TABLE) on the database.
    
    Args:
        query: The SQL query string
        parameters: Optional list of query parameters
    """
    logger.info(f"Executing write query: {query} with params: {parameters}")
    conn = get_db()
    try:
        cursor = conn.cursor()
        if parameters:
            cursor.execute(query, parameters)
        else:
            cursor.execute(query)
        conn.commit()
        affected = cursor.rowcount
        return f"Query executed successfully. Row count affected: {affected}"
    except Exception as e:
        logger.error(f"Database error executing query: {e}")
        return f"Error: {str(e)}"
    finally:
        conn.close()

@mcp.tool()
def read_query(query: str, parameters: Optional[List[Any]] = None) -> str:
    """
    Execute a read SQL query (SELECT) on the database.
    
    Args:
        query: The SELECT SQL query string
        parameters: Optional list of query parameters
    """
    logger.info(f"Executing read query: {query} with params: {parameters}")
    conn = get_db()
    try:
        cursor = conn.cursor()
        if parameters:
            cursor.execute(query, parameters)
        else:
            cursor.execute(query)
        rows = cursor.fetchall()
        result = [dict(row) for row in rows]
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Database error executing query: {e}")
        return json.dumps({"error": str(e)})
    finally:
        conn.close()

if __name__ == "__main__":
    mcp.run(transport="stdio")
