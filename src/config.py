import os
import sqlite3
import configparser
from sqlalchemy import create_engine

config = configparser.ConfigParser()
config.read(os.path.join('config','config.txt'))

engine = create_engine(config.get('database', 'con'))


