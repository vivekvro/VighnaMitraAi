import streamlit as st, re, datetime as dt,time,os
from uuid import uuid4
from src.encrypt import PasswordEncoder,ComparePasswords
from src.user_auth import (
    create_accounts_info_table,
    insert_account_info,
    check_if_email_exists,
    check_if_user_exists,
    fetch_password_by_username
    )
from langgraph.checkpoint.sqlite import SqliteSaver
from sqlite3 import connect
from src.chatbots.chatbot_graphs import base_chatbot
from langchain_core.messages import HumanMessage
from src.rag.DocumentsLoader import  load_tempfile_path
import requests
from dotenv import load_dotenv

load_dotenv()




db_path = "data/vighnamitraai.db"

URL_MAIN = os.getenv("MAIN_URL")
URL_upload_document = URL_MAIN+"/vm/upload_document"


chatbot = base_chatbot()










def create_timestamp(db_path):
    with connect(db_path) as con:
        cur = con.cursor()

        # ✅ check table exists first
        cur.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='checkpoints';
        """)
        if not cur.fetchone():
            return  # 🚀 skip safely

        # check columns
        cur.execute("PRAGMA table_info(checkpoints)")
        columns = [col[1] for col in cur.fetchall()]

        if "created_at" not in columns:
            cur.execute("""
                ALTER TABLE checkpoints 
                ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            """)






create_accounts_info_table(db_path=db_path)



create_timestamp(db_path)

def validate_username(username: str):
    pattern = r"^[a-zA-Z0-9._]{3,20}$"
    
    if not re.match(pattern, username):
        raise ValueError("Username must be 3-20 chars, only letters, numbers, '_' or '.' allowed")
    
    return True

def validate_email(email: str):
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    
    if not re.match(pattern, email):
        raise ValueError("Invalid email format")
    
    return True

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


def confirm_passwords(pwd,c_pwd):
    if pwd!=c_pwd:
        raise ValueError("Passwords do not match")
    return True







if "user" not in st.session_state:
    entrypoint = st.selectbox("SignUp/In",["New User","Existing User"])

    if entrypoint == "New User":
        st.subheader("Welcome")
        username = st.text_input("Username")
        dob = st.date_input(
            "date of birth",
            min_value=dt.date(1947,8,15),
            max_value=dt.date.today(),
            format="YYYY-MM-DD")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        if st.button("**SignUp**"):
            try:
                if check_if_user_exists(username,db_path):
                    st.error("User already exists!")
                    st.stop()
                if check_if_email_exists(email,db_path):
                    st.error("Email already exists!")
                    st.stop()
                validate_username(username=username)
                validate_email(email=email)
                if not validate_password(pw=password):
                    st.error("Password must contain:\n- Minimum 8 characters\n- At least 1 uppercase letter (A-Z)\n- At least 1 lowercase letter (a-z)\n- At least 1 number (0-9)\n- At least 1 special character (!@#$%^&*)")
                    st.stop()
                confirm_passwords(password,confirm_password)
                insert_account_info(username=username,password=password,email=email,dob=dob,db_path=db_path)
                st.success(f"Account Created! hey {username.lower().strip()}.")
                st.session_state['user']={"username":username.lower().strip()}
                st.rerun()



            except ValueError as e:
                st.error(e)
                st.stop()
                
            
            

    elif entrypoint == "Existing User":
        st.subheader("Welcome Back :-)")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("SignIn"):
                
            try:
                if not username or not password:
                    st.warning("Please fill all fields")
                    st.stop()
                if not check_if_user_exists(username,db_path):
                        st.error("User does not exist!")
                        st.stop()
                stored_pwd = fetch_password_by_username(username=username,db_path=db_path)
                if not ComparePasswords(password,stored_pwd):
                    st.error("Invalid Password")
                    st.stop()
                
                st.session_state['user'] = {"username":username.lower().strip()}
                st.rerun()
            except ValueError as e:
                st.error(e)
                st.rerun()
                
            
    st.stop()




def get_all_threads(db_path, user):
    if not os.path.exists(db_path):
        return []

    with connect(db_path) as con:
        cur = con.cursor()

        # check table exists
        cur.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='checkpoints';
        """)
        if not cur.fetchone():
            return []

        cur.execute("""
            SELECT DISTINCT thread_id
            FROM checkpoints
            WHERE thread_id LIKE ?
            ORDER BY created_at DESC
        """, (f"{user}%",))

        rows = cur.fetchall()

    return [r[0] for r in rows]


username = st.session_state['user']['username']

if "chat_id"  not in  st.session_state["user"]:
    st.session_state["user"]['chat_id']= f"{username}_{str(uuid4())}"


def is_chat_empty(thread_id):
    if not os.path.exists(db_path):
        return True

    with connect(db_path) as con:
        cur = con.cursor()

        # ensure table exists
        cur.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='checkpoints';
        """)
        if not cur.fetchone():
            return True

        cur.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id=?",
            (thread_id,)
        )
        count = cur.fetchone()[0]

    return count == 0


st.sidebar.title("Vighna Mitra Ai")
st.sidebar.markdown("---")
st.sidebar.header(f"Username: {username}")
st.sidebar.markdown("---")

if st.sidebar.button("New chat"):
    current_id = st.session_state["user"]["chat_id"]

    if is_chat_empty(current_id):
        st.sidebar.warning("Current chat is empty. Use it first.")
    else:
        st.session_state["user"]["chat_id"] = f"{username}_{str(uuid4())}"
        st.rerun()






st.sidebar.markdown("---\ncurrent chat:")
st.sidebar.button(st.session_state["user"]["chat_id"],width=200)
st.sidebar.markdown("---\n")
sidebar_sections = st.sidebar.selectbox("select: ",["chat history","connectors","attach documents"],width=200)
if sidebar_sections =="chat history":
    chat_history = get_all_threads(db_path,username)
    st.sidebar.markdown("---\nchat history chat:")
    if chat_history:
        for chat in chat_history:
            if st.sidebar.button(f"{chat}"):
                st.session_state["user"]["chat_id"] = chat
                st.rerun()

elif sidebar_sections == "attach documents":
    doctype = st.sidebar.selectbox(
        "select document type", ["pdf", "txt", "url"], width=200
    )

    file_details = None  # initialize

    if doctype in ["pdf", "txt"]:
        uploaded = st.sidebar.file_uploader(
            "upload here", type=["pdf", "txt"], width=200
        )
        
        if uploaded is not None:
            path = load_tempfile_path(uploaded)
            file_details = {
                "path": path,
                "doctype": doctype,
                "user_id": username
            }
            

    else:
        path = st.sidebar.text_input("Enter URL:", width=200)
        
        if path:
            file_details = {
                "path": path,
                "doctype": doctype,
                "user_id": username
            }
    if st.sidebar.button("Upload"):
        if file_details is None:
            st.sidebar.warning("Please upload a file or enter a URL first ⚠️")
        else:
            with st.sidebar.spinner("Uploading..."):

                try:
                    if doctype in ["pdf", "txt"]:

                        if uploaded is not None:
                            path = load_tempfile_path(uploaded)
                            file_details = {
                                "path": path,
                                "doctype": doctype,
                                "user_id": username
                            }
                    response = requests.post(
                        url=URL_upload_document,
                        json=file_details,
                    )
                    result = response.json()
                    st.sidebar.success(result['response'])
                except Exception as e:
                    st.sidebar.error(str(e))



elif sidebar_sections == "connectors":
    connectors_type = st.sidebar.selectbox("select MCP Server  type :",["online","local"])
    if connectors_type == "online":
        mcp_server_name= st.sidebar.text_input("server Name:")
        mcp_server_url = st.sidebar.text_input("server url:")
        st.session_state['user']["mcp"] = {
            "type":connectors_type,
            "server_info":{
                "name":mcp_server_name,
                "url":mcp_server_url
                }
                }
    if connectors_type =="local":
        if st.sidebar.button("Coming Soon."):
            with st.sidebar.spinner("......"):
                time.sleep(10)
                st.sidebar.success("Still you need to wait.")





st.sidebar.markdown("---")
if st.sidebar.button("logout"):
    if "user" in st.session_state:
        del st.session_state["user"]
    st.rerun()

config = {"configurable":{
            "user_id":username,
            "thread_id":st.session_state['user']['chat_id']
        }
    }



def get_messages(config):
    state = chatbot.get_state(config=config)
    return state.values.get("messages",[])



for msg in get_messages(config):
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.write(msg.content)


user_input = st.chat_input("Ask Anything")



def fake_stream_response(text:str):
    for char in text:
        yield char
        time.sleep(0.002)


if user_input:
    with st.chat_message(name="user"):
        st.write(user_input)
    with st.spinner("thinking...."):
        result_state = chatbot.invoke({
            "messages":[HumanMessage(content=user_input)],
            "trace":[]
            },config=config)

        

    with st.chat_message(name="assistant"):
        if "trace" in result_state:# 👇 Process tracer
            with st.expander("⚙️ Execution Trace"):
                for step in result_state["trace"]:
                    st.write(step)
        st.write_stream(fake_stream_response(result_state['messages'][-1].content))

