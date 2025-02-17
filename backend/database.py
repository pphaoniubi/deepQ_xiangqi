import pymysql

# MySQL Database Configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "pphaoniubi",
    "password": "12345678pP!",
    "database": "xiangqi_db",
    "cursorclass": pymysql.cursors.DictCursor  # Returns results as dictionaries
}

# Function to get a new database connection
def get_db_connection():
    return pymysql.connect(**DB_CONFIG)
