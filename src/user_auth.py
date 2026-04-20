import sqlite3
import re
from src.encrypt import PasswordEncoder,ComparePasswords
from textwrap import dedent







#------------------------ Create Table -------------------------------

def create_accounts_info_table(db_path:str):
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS accounts_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password  BLOB,
                    dob DATE,
                    email TEXT Unique
                );
                    """)
        con.commit()


#------------------------ password -------------------------------



# validate

def validate_password(pw):
    if len(pw) < 8:
        return False
    if not re.search(r"[A-Z]", pw):
        return False
    if not re.search(r"[a-z]", pw):
        return False
    if not re.search(r"\d", pw):
        return False
    if not re.search(r"[!@#$%^&*]", pw):
        return False
    return True



# fetch


def fetch_password_by_username(username:str,db_path:str):
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute("SELECT password from accounts_info where username = ?",(username,))
        row = cur.fetchone()
    return row[0] if row else None


#------------------------ check if Xyz exists-----------------
def check_if_user_exists(username:str,db_path:str):
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute("SELECT username from accounts_info where username = ?",(username,))
        row = cur.fetchone()
    return row is not None

def check_if_email_exists(email:str,db_path:str):
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute("SELECT email from accounts_info where email = ?",(email,))
        row = cur.fetchone()
    return row is not None


#------------------------ Signup -------------------------------

def insert_account_info(username:str,password:str,dob:str,email:str,db_path):
    if not all([username,password,dob,email]):
        raise ValueError(dedent("All the fields are required"))
    username= username.lower().strip()
    email = email.lower().strip()
    encoded_pwd = PasswordEncoder(password=password)
    try:
        with sqlite3.connect(db_path) as con:
            cur = con.cursor()
            cur.execute("""INSERT INTO accounts_info
                                (username, password, dob, email) VALUES (?, ?, ?, ?);
                        """,(username, encoded_pwd, dob, email))
            con.commit()
    except sqlite3.IntegrityError:
        raise ValueError(dedent("username or email already exists"))




#------------------------ Login-------------------------------
def login_account(username:str,password:str,db_path:str):
    if not username or  not password:
        raise ValueError(dedent("username and password required"))
    stored_pwd = fetch_password_by_username(username=username,db_path=db_path)
    if stored_pwd is None:
            raise ValueError(dedent("user does not exist, please signup first!"))
    status = ComparePasswords(login_password=password,stored_hashed_password=stored_pwd)
    return status
