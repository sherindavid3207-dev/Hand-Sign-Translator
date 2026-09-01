import sqlite3
def init_db():
    conn= sqlite3.connect("hand_dataset.db")
    cursor = conn.cursor()
    columns = [f"lm_{i}" for i in range(63)]
    columns_sql = ", ".join([f"{col} REAL" for col in columns])
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS landmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        {columns_sql},
        label TEXT NOT NULL
    )
    """
    cursor.execute(create_table_query)
    conn.commit()
    conn.close()
    
    print("Database 'hand_dataset.db' initialized with 'landmarks' table successfully!")

if __name__ == "__main__":
    init_db()
