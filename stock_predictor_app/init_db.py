
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.database import Base, engine

def init_db():
    # Drop all existing tables
    Base.metadata.drop_all(engine)
    # Create all tables
    Base.metadata.create_all(engine)
    print("Database initialized successfully!")

if __name__ == '__main__':
    init_db()
