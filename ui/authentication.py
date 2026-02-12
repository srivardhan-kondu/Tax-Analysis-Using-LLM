"""
Authentication module for Tax Evasion Detection System
Full user management with registration, login/logout, and role-based access
"""
import streamlit as st
import hashlib
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

# User database file
USER_DB_FILE = Path(__file__).parent.parent / "data" / "users.json"


def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def load_users() -> Dict:
    """Load users from file"""
    if USER_DB_FILE.exists():
        with open(USER_DB_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_users(users: Dict):
    """Save users to file"""
    USER_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USER_DB_FILE, 'w') as f:
        json.dump(users, f, indent=2)


def initialize_default_users():
    """Initialize with default admin user"""
    users = load_users()
    if not users or 'admin' not in users:
        if not users:
            users = {}
        users["admin"] = {
            "password": hash_password("admin123"),
            "role": "admin",
            "name": "Administrator",
            "analysis_results": []
        }
        save_users(users)
    return users


def register_user(username: str, password: str, name: str) -> tuple:
    """
    Register a new user
    Returns: (success: bool, message: str)
    """
    if not username or not password or not name:
        return False, "All fields are required"
    
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    
    users = load_users()
    
    if username.lower() in [u.lower() for u in users.keys()]:
        return False, "Username already exists"
    
    # Create new user with 'user' role
    users[username] = {
        "password": hash_password(password),
        "role": "user",
        "name": name,
        "analysis_results": [],
        "created_at": datetime.now().isoformat()
    }
    
    save_users(users)
    return True, f"Account created successfully! Welcome, {name}!"


def verify_credentials(username: str, password: str) -> bool:
    """Verify user credentials"""
    users = load_users()
    if username in users:
        hashed_input = hash_password(password)
        return users[username]["password"] == hashed_input
    return False


def get_user_info(username: str) -> Dict:
    """Get user information"""
    users = load_users()
    return users.get(username, {})


def login_user(username: str):
    """Log in a user by setting session state"""
    st.session_state['logged_in'] = True
    st.session_state['username'] = username
    user_info = get_user_info(username)
    st.session_state['user_role'] = user_info.get('role', 'user')
    st.session_state['user_name'] = user_info.get('name', username)
    
    # Load user's analysis results
    st.session_state['analysis_results'] = load_user_analysis(username)


def logout_user():
    """Log out current user"""
    keys_to_clear = ['logged_in', 'username', 'user_role', 'user_name', 
                     'analysis_results', 'current_data', 'manual_entry_data',
                     'trigger_manual_analysis']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def is_logged_in() -> bool:
    """Check if user is logged in"""
    return st.session_state.get('logged_in', False)


def is_admin() -> bool:
    """Check if current user is admin"""
    return st.session_state.get('user_role') == 'admin'


def require_login():
    """Require login to access feature"""
    if not is_logged_in():
        st.warning("⚠️ Please log in to access this feature")
        st.stop()


# ============ User Analysis Data Functions ============

def save_user_analysis(username: str, results_df: pd.DataFrame):
    """Save analysis results for a specific user"""
    users = load_users()
    
    if username not in users:
        return
    
    # Ensure analysis_results field exists
    if 'analysis_results' not in users[username]:
        users[username]['analysis_results'] = []
    
    # Convert DataFrame to list of dicts
    results_list = results_df.to_dict('records')
    
    # Add timestamp to each record
    timestamp = datetime.now().isoformat()
    for record in results_list:
        record['saved_at'] = timestamp
        # Handle non-serializable types
        for key, value in record.items():
            if isinstance(value, (list, dict)):
                record[key] = value
            elif pd.isna(value):
                record[key] = None
    
    # Append new results (not replace)
    users[username]['analysis_results'].extend(results_list)
    
    save_users(users)


def load_user_analysis(username: str) -> Optional[pd.DataFrame]:
    """Load analysis results for a specific user"""
    users = load_users()
    
    if username not in users:
        return None
    
    results = users[username].get('analysis_results', [])
    
    if not results:
        return None
    
    return pd.DataFrame(results)


def load_all_users_analysis() -> Optional[pd.DataFrame]:
    """Load all users' analysis results (admin only)"""
    users = load_users()
    
    all_results = []
    for username, user_data in users.items():
        user_results = user_data.get('analysis_results', [])
        for result in user_results:
            result['analyzed_by'] = username
            all_results.append(result)
    
    if not all_results:
        return None
    
    return pd.DataFrame(all_results)


def get_all_usernames() -> List[str]:
    """Get list of all usernames (admin only)"""
    users = load_users()
    return list(users.keys())


def clear_user_analysis(username: str):
    """Clear all analysis results for a user"""
    users = load_users()
    if username in users:
        users[username]['analysis_results'] = []
        save_users(users)


# ============ UI Functions ============

def show_login_page():
    """Display login/register page"""
    st.title("🔐 Tax Evasion Detection System")
    
    # Initialize default users
    initialize_default_users()
    
    # Tabs for Login and Register
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Create Account"])
    
    with tab1:
        st.subheader("User Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", type="primary")
            
            if submit:
                if verify_credentials(username, password):
                    login_user(username)
                    st.success(f"✅ Welcome back, {st.session_state.get('user_name', username)}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
    
    with tab2:
        st.subheader("Create New Account")
        with st.form("register_form"):
            new_username = st.text_input("Choose Username", key="reg_username")
            new_name = st.text_input("Your Full Name", key="reg_name")
            new_password = st.text_input("Choose Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
            
            register_btn = st.form_submit_button("Create Account", type="primary")
            
            if register_btn:
                if new_password != confirm_password:
                    st.error("❌ Passwords do not match")
                else:
                    success, message = register_user(new_username, new_password, new_name)
                    if success:
                        st.success(f"✅ {message}")
                        st.info("👆 Now switch to the Login tab to sign in!")
                    else:
                        st.error(f"❌ {message}")
    
    # Show demo credentials
    with st.expander("🔑 Demo Admin Credentials"):
        st.info("**Admin:** `admin` / `admin123`")


def show_user_info():
    """Display logged-in user information with logout button"""
    if is_logged_in():
        user_name = st.session_state.get('user_name', st.session_state.get('username'))
        user_role = st.session_state.get('user_role', 'user')
        
        # Role badge
        role_badge = "👑 Admin" if user_role == 'admin' else "👤 User"
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{user_name}** ({role_badge})")
        with col2:
            if st.button("Logout", key="logout_btn", type="secondary"):
                logout_user()
                st.rerun()
# Authentication Module
