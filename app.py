import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta, time 
import plotly.express as px
import plotly.graph_objects as go
import io
import math
import json
from typing import List, Dict, Optional, Tuple

# --- Configuration ---
DISPLAY_TIMEZONE = 'Europe/Brussels'
# Page size is 1000 for better reliability and responsiveness
PAGE_SIZE = 1000 
REQUEST_TIMEOUT = 90 # seconds

# ---- Project History: all actions & per-action fields (unchanged) ----
PROJECT_HISTORY_ACTIONS = [
    "add-media",
    "add-submission",
    "allow-anonymous-submissions",
    "archive",
    "clone-permissions",
    "connect-project",
    "delete-media",
    "delete-service",
    "delete-submission",
    "deploy",
    "disable-sharing",
    "disallow-anonymous-submissions",
    "disconnect-project",
    "enable-sharing",
    "export",
    "modify-imported-fields",
    "modify-qa-data",
    "modify-service",
    "modify-sharing",
    "modify-submission",
    "modify-user-permissions",
    "redeploy",
    "register-service",
    "replace-form",
    "share-data-publicly",
    "share-form-publicly",
    "transfer",
    "unarchive",
    "unshare-data-publicly",
    "unshare-form-publicly",
    "update-content",
    "update-name",
    "update-settings",
    "update-qa",
]

# Per-action filterable fields for Project History Logs (unchanged)
ACTION_FIELD_MAP: Dict[str, List[Tuple[str, str]]] = {
    "add-media": [
        ("metadata__asset-file__uid", "Asset File UID"),
        ("metadata__asset-file__filename", "Asset File Filename"),
    ],
    "add-submission": [
        ("metadata__submission__submitted_by", "Submission Submitted By"),
        ("metadata__submission__root_uuid", "Submission Root UUID"),
    ],
    "archive": [
        ("metadata__latest_version_uid", "Latest Version UID"),
    ],
    "clone-permissions": [
        ("metadata__cloned_from", "Cloned From"),
    ],
    "connect-project": [
        ("metadata__paired-data__source_uid", "Paired Data Source UID"),
        ("metadata__paired-data__source_name", "Paired Data Source Name"),
    ],
    "delete-media": [
        ("metadata__asset-file__uid", "Asset File UID"),
        ("metadata__asset-file__filename", "Asset File Filename"),
    ],
    "delete-service": [
        ("metadata__hook__uid", "Hook UID"),
        ("metadata__hook__endpoint", "Hook Endpoint"),
        ("metadata__hook__active", "Hook Active (true/false)"),
    ],
    "delete-submission": [
        ("metadata__submission__submitted_by", "Submission Submitted By"),
        ("metadata__submission__root_uuid", "Submission Root UUID"),
    ],
    "deploy": [], 
    "disconnect-project": [
        ("metadata__paired-data__source_uid", "Paired Data Source UID"),
        ("metadata__paired-data__source_name", "Paired Data Source Name"),
    ],
    "modify-imported-fields": [
        ("metadata__paired-data__source_uid", "Paired Data Source UID"),
        ("metadata__paired-data__source_name", "Paired Data Source Name"),
    ],
    "modify-qa-data": [
        ("metadata__submission__submitted_by", "Submission Submitted By"),
        ("metadata__submission__root_uuid", "Submission Root UUID"),
    ],
    "modify-service": [
        ("metadata__hook__uid", "Hook UID"),
        ("metadata__hook__endpoint", "Hook Endpoint"),
        ("metadata__hook__active", "Hook Active (true/false)"),
    ],
    "modify-submission": [
        ("metadata__submission__submitted_by", "Submission Submitted By"),
        ("metadata__submission__root_uuid", "Submission Root UUID"),
        ("metadata__submission__status", "Submission Status (if changed)"),
    ],
    "modify-user-permissions": [
        ("metadata__permissions__username", "Permissions Username"),
    ],
    "redeploy": [], 
    "register-service": [
        ("metadata__hook__uid", "Hook UID"),
        ("metadata__hook__endpoint", "Hook Endpoint"),
        ("metadata__hook__active", "Hook Active (true/false)"),
    ],
    "transfer": [
        ("metadata__username", "Username (Assignee)"),
    ],
    "unarchive": [
        ("metadata__latest_version_uid", "Latest Version UID"),
    ],
    "update-name": [
        ("metadata__name__old", "Name Old"),
        ("metadata__name__new", "Name New"),
    ],
    "update-settings": [
        ("metadata__settings__description__old", "Settings Description Old"),
        ("metadata__settings__description__new", "Settings Description New"),
    ],
    # Actions with no extra fields beyond common filters:
    "allow-anonymous-submissions": [],
    "disable-sharing": [],
    "disallow-anonymous-submissions": [],
    "enable-sharing": [],
    "export": [],
    "modify-sharing": [],
    "replace-form": [],
    "share-data-publicly": [],
    "share-form-publicly": [],
    "unshare-data-publicly": [],
    "unshare-form-publicly": [],
    "update-content": [],
    "update-qa": [],
}

# API Endpoints Configuration with comprehensive filters (unchanged)
API_ENDPOINTS = {
    # Audit Logs (NO count, uses 'next' for pagination)
    "Audit Logs": {
        "path": "/api/v2/audit-logs/",
        "filters": {
            "actions": [], 
            "log_types": [
                "access", "project-history", "data-editing",
                "submission-management", "user-management", "asset-management"
            ],
            "metadata_fields": [
                ("metadata__ip_address", "IP Address")
            ]
        },
        "offset_pagination": False # DRF style pagination (next/previous)
    },
    # Project History Logs (HAS count in first response, uses offset for pagination)
    "Project History Logs": {
        "path": "/api/v2/project-history-logs/",
        "filters": {
            "actions": PROJECT_HISTORY_ACTIONS,
            "log_types": [],
            "metadata_fields": []
        },
        "offset_pagination": True, # Offset/Limit style pagination (with count)
        "common_fields": [
            ("user_uid", "User UID"),
            ("user__email", "User Email"),
            ("user__is_superuser", "Is Superuser (true/false)"),
            ("metadata__source", "Metadata Source"),
            ("metadata__ip_address", "IP Address"),
            ("metadata__asset_uid", "Asset UID"),
            ("metadata__log_subtype", "Log Subtype (project/permission)"),
            ("action", "Action") # This is handled by multiselect
        ]
    },
    # Access Logs (NO count, uses 'next' for pagination)
    "Access Logs": {
        "path": "/api/v2/access-logs/",
        "filters": {
            "actions": [],
            "log_types": [],
            # Access Log specific metadata fields
            "metadata_fields": [
                ("metadata__source", "Source (Browser/App)"),
                ("metadata__auth_type", "Auth Type (Token/Basic)"),
                ("metadata__ip_address", "IP Address")
            ]
        },
        "offset_pagination": False # DRF style pagination (next/previous)
    }
}


# Date range presets (unchanged)
DATE_PRESETS = {
    "Today": 0,
    "Last 7 days": 7,
    "Last 14 days": 14,
    "Last 30 days": 30,
    "Last 90 days": 90,
    "Last 6 months": 180,
    "Last year": 365,
    "Custom": None
}

# --- Helper Functions (Updated) ---
def validate_server_url(url: str) -> Tuple[bool, str]:
    url = url.strip().rstrip('/')
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        return True, base_url
    except Exception as e:
        return False, f"Invalid URL: {str(e)}"

def _normalize_value_for_query(value: str) -> str:
    """Normalize boolean-ish strings and quote values containing spaces."""
    if isinstance(value, bool):
        return "true" if value else "false"
    s = str(value).strip()
    if s.lower() in {"true", "false"}:
        return s.lower()
    # quote if contains whitespace or colon
    if any(ch in s for ch in [' ', ':']):
        return f'"{s}"'
    return s

def build_query(
    selected_actions: List[str],
    selected_log_types: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    custom_query: str,
    username: str = "",
    asset_uid: str = "",
    extra_filters: Optional[Dict[str, str]] = None
) -> str:
    """Build query string from filters following KoboToolbox API format."""
    query_parts = []

    # Actions
    if selected_actions:
        if len(selected_actions) == 1:
            query_parts.append(f"action:{selected_actions[0]}")
        else:
            action_query = " OR ".join([f"action:{a}" for a in selected_actions])
            query_parts.append(f"({action_query})")

    # Log types
    if selected_log_types:
        if len(selected_log_types) == 1:
            query_parts.append(f"log_type:{selected_log_types[0]}")
        else:
            log_type_query = " OR ".join([f"log_type:{t}" for t in selected_log_types])
            query_parts.append(f"({log_type_query})")

    # Date and Time filters (STRICTLY DATE-BASED)
    
    # 1. GTE Filter (Start Date - 00:00)
    if start_date is not None:
        # Start at 00:00 on the selected start date
        start_datetime_str = f"{start_date.strftime('%Y-%m-%d')} 00:00"
        query_parts.append(f'date_created__gte:"{start_datetime_str}"')
    
    # 2. LT Filter (End Date - Full Day Coverage)
    if end_date is not None:
        # Advance the end date by one day and set time to 00:00.
        # This ensures all logs up to 23:59:59 on the selected end_date are included.
        end_datetime_boundary = end_date + timedelta(days=1)
        end_datetime_str = f"{end_datetime_boundary.strftime('%Y-%m-%d')} 00:00"
        query_parts.append(f'date_created__lt:"{end_datetime_str}"')


    # Legacy convenience filters (only username is always checked)
    if username:
        query_parts.append(f"user__username:{_normalize_value_for_query(username)}")
    if asset_uid:
        query_parts.append(f"metadata__asset_uid:{_normalize_value_for_query(asset_uid)}")

    # Extra filters (common + per-action + access logs metadata)
    if extra_filters:
        for field, value in extra_filters.items():
            if value is None:
                continue
            s = str(value).strip()
            if s == "":
                continue
            query_parts.append(f"{field}:{_normalize_value_for_query(s)}")

    # Custom (verbatim)
    if custom_query:
        query_parts.append(custom_query)

    return " AND ".join(query_parts) if query_parts else ""

def get_api_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

def fetch_logs_paginated(
    api_url: str,
    api_key: str,
    query_string: str,
    use_offset_pagination: bool
) -> Tuple[List[Dict], Optional[int]]:
    """
    Fetch all logs with pagination and progress tracking. 
    Returns logs and final count (if known/fetched).
    """
    headers = get_api_headers(api_key)
    logs: List[Dict] = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_count: Optional[int] = None 

    try:
        if use_offset_pagination:
            # === Project History Logs (Offset Pagination with COUNT in first response) ===
            
            # 1. Fetch the first full page (offset 0) to get the reliable count and initial results
            status_text.text(f"Fetching first page (limit {PAGE_SIZE:,}) to determine total logs...")
            params = {"format": "json", "limit": PAGE_SIZE, "offset": 0}
            if query_string:
                params["q"] = query_string
            
            response = requests.get(api_url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            # Use the count from the first response
            total_count = data.get("count", 0) 
            logs.extend(data.get("results", []))
            
            if total_count == 0:
                progress_bar.empty()
                status_text.empty()
                return [], 0 
            
            total_pages = math.ceil(total_count / PAGE_SIZE)

            if total_count > PAGE_SIZE:
                 st.warning(f"⚠️ Found {total_count:,} logs. Fetching will require {total_pages} requests. This may take some time.")
            
            progress_bar.progress(min(1.0, 1 / total_pages)) # Initial progress for first page

            # 2. Continue fetching remaining pages (starting from page index 1)
            for page in range(1, total_pages):
                offset = page * PAGE_SIZE
                status_text.text(f"Fetching page {page + 1}/{total_pages} "
                                    f"(offset {offset:,}, {len(logs):,} logs so far)...")
                params = {"format": "json", "limit": PAGE_SIZE, "offset": offset}
                if query_string:
                    params["q"] = query_string
                response = requests.get(api_url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                data = response.json()
                logs.extend(data.get("results", []))
                progress_bar.progress((page + 1) / total_pages)
        else:
            # === Audit & Access Logs (DRF Next/Previous Pagination - NO COUNT) ===
            next_url: Optional[str] = api_url
            params = {"format": "json", "limit": PAGE_SIZE}
            if query_string:
                params["q"] = query_string
            
            page_index = 0
            while next_url:
                page_index += 1
                status_text.text(f"Fetching page {page_index} ({len(logs):,} logs so far)...")
                
                # First request uses params, subsequent requests use the 'next' URL directly
                if page_index == 1:
                     response = requests.get(next_url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
                else:
                     response = requests.get(next_url, headers=headers, timeout=REQUEST_TIMEOUT)

                response.raise_for_status()
                data = response.json()
                page_results = data.get("results", [])
                logs.extend(page_results)
                next_url = data.get("next")
                
                # Show progress based on pages fetched (since total is unknown)
                progress_bar.progress(min(0.99, page_index * 0.01)) 
            
            # Since count is unknown, set it to the total number of records fetched
            total_count = len(logs) 


        progress_bar.empty()
        status_text.empty()
        return logs, total_count

    except requests.exceptions.Timeout:
        progress_bar.empty()
        status_text.empty()
        raise Exception("Request timed out while fetching logs.")
    except requests.exceptions.RequestException as e:
        progress_bar.empty()
        status_text.empty()
        # Explicitly check for 403 or 401 which may look like 'no logs found'
        if 'response' in locals() and response.status_code in [401, 403]:
             raise Exception(f"API Error: Authentication failed (Status {response.status_code}). Check your API Token and permissions.")
        raise Exception(f"Error fetching logs: {str(e)}")

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    date_fields = ['date_created', 'timestamp', 'created_at', 'date']
    date_field = next((f for f in date_fields if f in df.columns), None)
    if date_field:
        df[date_field] = (
            pd.to_datetime(df[date_field], utc=True)
            .dt.tz_convert(DISPLAY_TIMEZONE)
        )
        df = df.sort_values(date_field, ascending=False)
    return df

def create_export_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    export_df = df.copy()
    for col in export_df.select_dtypes(include=["datetimetz"]).columns:
        # Convert timezone aware column to naive timestamp before export
        export_df[col] = export_df[col].dt.tz_localize(None) 
    return export_df

def get_date_column(df: pd.DataFrame) -> Optional[str]:
    for field in ['date_created', 'timestamp', 'created_at', 'date']:
        if field in df.columns:
            return field
    return None

def build_full_url(base_url: str, endpoint_path: str, query_string: str) -> str:
    full_url = base_url + endpoint_path + f"?format=json&limit={PAGE_SIZE}"
    if query_string:
        full_url += f"&q={query_string}"
    return full_url

# --- Streamlit Layout ---
st.set_page_config(
    page_title="KoboToolbox Logs Viewer",
    page_icon="🕵️",
    layout="wide"
)

st.title("🕵️ KoboToolbox Logs Viewer & Analyzer")
st.markdown(
    "Fetch, explore, visualize, and export logs from your KoboToolbox instance. "
    "Supports **Audit Logs**, **Project History**, and **Access Logs**."
)

# Sidebar inputs
with st.sidebar:
    st.header("🌐 Server URL & 🔑 Authentication")
    default_url = "eu.kobotoolbox.org"
    server_url_input = st.text_input(
        "Server URL:",
        value=default_url,
        help="Enter your KoboToolbox server URL"
    )

    if server_url_input:
        is_valid, result = validate_server_url(server_url_input)
        if is_valid:
            server_url = result
        else:
            st.error(f"❌ {result}")
            server_url = None
    else:
        st.warning("⚠️ Please enter a server URL")
        server_url = None


    # API Key
    api_key = st.text_input(
        "API Token:",
        type="password",
        help="Enter your KoboToolbox API token"
    )

    # API Endpoint Selection
    st.header("📡 API Endpoint")
    selected_endpoint = st.selectbox(
        "Select Endpoint:",
        options=list(API_ENDPOINTS.keys()),
        help="Choose which type of logs to retrieve"
    )

    endpoint_config = API_ENDPOINTS[selected_endpoint]


    # Date Range with presets
    st.header("⚙️ Filters")

    date_preset = st.selectbox(
        "Date range:",
        options=list(DATE_PRESETS.keys()),
        index=1 # Default to "Last 7 days"
    )

    col_date1, col_date2 = st.columns(2)
    

    if date_preset == "Custom":
        with col_date1:
            start_date = st.date_input("From Date", value=(datetime.today() - timedelta(days=7)).date())
        with col_date2:
            end_date = st.date_input("To Date", value=datetime.today().date())
    else:
        days_back = DATE_PRESETS[date_preset]
        start_date = (datetime.today() - timedelta(days=days_back)).date()
        end_date = datetime.today().date()
        st.caption(f"📆 Default Date Range: {start_date} to {end_date}")

    

    # --- INPUT VALIDATION ---
    start_datetime_check = datetime.combine(start_date, time(0, 0))
    
    # Validation checks against the original intent (end of day on end_date)
    end_datetime_boundary = datetime.combine(end_date, time(0, 0)) + timedelta(days=1)

    if start_datetime_check >= end_datetime_boundary:
        st.error("❌ Invalid date range: Start date must be before or equal to the End Date.")
        st.stop()


    # Initialize all filter lists/dictionaries
    selected_actions: List[str] = []
    selected_log_types: List[str] = []
    extra_filters: Dict[str, str] = {}
    
    # 1. Action Type Selection (Dynamically shown ONLY for Project History)
    actions_list = endpoint_config['filters'].get('actions')
    if selected_endpoint == "Project History Logs" and actions_list:
        st.subheader("Action Type")
        selected_actions = st.multiselect(
            "Action(s):",
            actions_list,
            default=[], 
            help="Filter by specific history actions"
        )

    # 2. Log Type Selection (Only for Audit Logs)
    if selected_endpoint == "Audit Logs" and endpoint_config['filters']['log_types']:
        st.subheader("Log Type")
        selected_log_types = st.multiselect(
            "Log Category:",
            endpoint_config['filters']['log_types'],
            default=[], # Defaults to nothing selected, as requested
            help="Filter by log category"
        )

    # 3. Username filter (relevant for all)
    username_filter = st.text_input(
        "Username:",
        "",
        help="Filter by specific username (user__username)",
        placeholder="e.g., john.doe"
    )

    # 4. Asset UID filter (Only relevant for Audit Logs and Project History Logs)
    asset_uid_filter = ""
    if selected_endpoint in ["Audit Logs", "Project History Logs"]:
        asset_uid_filter = st.text_input(
            "Asset UID (Project UID):",
            "",
            help="Filter by asset metadata UID (metadata__asset_uid)",
            placeholder="e.g., aBcDeFgHiJkLmNo"
        )
    
    # 5. Endpoint Specific Metadata Fields (Access Logs or Audit Logs)
    metadata_fields_to_show = endpoint_config['filters'].get("metadata_fields", [])
    if metadata_fields_to_show:
        # Use a more generic heading for the metadata fields
        st.subheader(f"{selected_endpoint} Details Filters")

        for field_name, label in metadata_fields_to_show:
            placeholder = ""
            if field_name == "metadata__auth_type":
                placeholder = "token or basic"
            elif field_name == "metadata__ip_address":
                placeholder = "94.111.54.172"
            elif field_name == "metadata__source":
                placeholder = "kpi or kobocat"
            
            # Audit Logs: Hide metadata__source if Audit Logs is selected
            if selected_endpoint == "Audit Logs" and field_name == "metadata__source":
                continue 

            val = st.text_input(label, value="", placeholder=placeholder, key=f"metadata_{field_name}")
            if val.strip():
                extra_filters[field_name] = val.strip()

    # 6. Dynamic/Extra Filters (Project History Specific - Common Fields)
    if selected_endpoint == "Project History Logs":
        st.subheader("Other Common Filters")
        
        # Define fields to SKIP based on the user's request
        FIELDS_TO_SKIP = [
            "action", "user__username", "metadata__asset_uid", 
            "metadata__source", "metadata__ip_address", 
            "user__email", "user__is_superuser", "metadata__log_subtype",
            "user_uid" # REMOVED user_uid as requested
        ]
        
        # Filter down common fields to just the ones we want to keep 
        fields_to_keep = [
            (f_name, f_label) for f_name, f_label in endpoint_config.get("common_fields", []) 
            if f_name not in FIELDS_TO_SKIP
        ]

        for field_name, label in fields_to_keep:
            placeholder = ""
            val = st.text_input(label, value="", placeholder=placeholder, key=f"common_{field_name}")
            if val.strip():
                extra_filters[field_name] = val.strip()

        # 7. Action-specific Fields (Project History Specific - ONLY display fields from ACTION_FIELD_MAP)
        if len(selected_actions) == 1:
            action_key = selected_actions[0]
            fields_for_action = ACTION_FIELD_MAP.get(action_key, [])
            if fields_for_action:
                st.subheader("Action-specific Parameters")
                for f_name, f_label in fields_for_action:
                    ph = "true or false" if f_name.endswith("__active") else ""
                    val = st.text_input(f_label, value="", placeholder=ph, key=f"act_{f_name}")
                    if val.strip():
                        extra_filters[f_name] = val.strip()
    
    # 8. Custom Query (Always shown)
    custom_query = st.text_input(
        "Custom Query:",
        "",
        help="Advanced query syntax (e.g., date_created__lt:\"2025-10-01\")",
        placeholder="field:value"
    )

    st.markdown("---")
    st.caption(f"🕐 Timezone: {DISPLAY_TIMEZONE}")
    st.caption("📖 API docs: kf.beta.kbtdev.org/api/v2/docs/")


# --- Main Logic ---

# Build full API URL
api_url = server_url + endpoint_config['path']

# Build query
start_date_ts = pd.to_datetime(start_date)
end_date_ts = pd.to_datetime(end_date)
final_query = build_query(
    selected_actions=selected_actions,
    selected_log_types=selected_log_types,
    start_date=start_date_ts,
    end_date=end_date_ts,
    custom_query=custom_query,
    username=username_filter,
    asset_uid=asset_uid_filter,
    extra_filters=extra_filters
)

# Display full URL
with st.expander("🔍 View Full API Request URL", expanded=False):
    full_request_url = build_full_url(server_url, endpoint_config['path'], final_query)
    st.code(full_request_url, language="text")
    st.caption("Copy this URL to test in your browser or API client (add your token header)")

# --- Data Fetching Button (MAIN SCREEN) ---
st.markdown("### 🚀 Fetch Data")

# Create a container for status messages, so we can clear/update them easily
status_placeholder = st.empty()

is_offset_pagination = endpoint_config.get("offset_pagination", False)

# Fetch button logic
if st.button("📥 Fetch All Logs", type="primary", use_container_width=True):
    try:
        status_placeholder.empty() # Clear previous status
        
        # Clear previous session data logs
        st.session_state.pop("df_logs", None) 
        st.session_state.pop("raw_logs", None)

        # Recalculate query (in case filters changed since last run)
        current_final_query = build_query(
            selected_actions=selected_actions,
            selected_log_types=selected_log_types,
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date),
            custom_query=custom_query,
            username=username_filter,
            asset_uid=asset_uid_filter,
            extra_filters=extra_filters
        )

        st.info(f"Fetching logs with page size {PAGE_SIZE:,}...")


        logs, total_count = fetch_logs_paginated(
            api_url, 
            api_key,
            current_final_query,
            use_offset_pagination=is_offset_pagination
        )

        if not logs:
            status_placeholder.warning("⚠️ No logs retrieved.")
            st.stop()

        status_placeholder.success(f"✅ Retrieved {len(logs):,} logs")

        with st.spinner("Processing for analysis..."):
            df_logs = pd.json_normalize(logs, sep='.')
            df_logs = process_dataframe(df_logs)
            st.session_state["df_logs"] = df_logs
            st.session_state["raw_logs"] = logs
            st.session_state["endpoint_name"] = selected_endpoint # Store name for analysis headers
            
            # Clear fetching state variables
            st.session_state.pop("total_count", None)
            st.session_state.pop("count_status_message", None)
            st.session_state.pop("query_string", None)
            st.session_state.pop("api_url", None)

    except Exception as e:
        status_placeholder.error(f"❌ Error during fetch: {e}")
        st.stop()


# --- Main Logic (Data Analysis and Visualization) ---

# Display and analyze data
if "df_logs" in st.session_state:
    df_logs = st.session_state["df_logs"]
    raw_logs = st.session_state["raw_logs"]
    endpoint_name = st.session_state.get("endpoint_name", selected_endpoint) # Use selected_endpoint if not in state

    st.markdown("---")
    st.markdown(f"## 📊 {endpoint_name} - Data Analysis")
    
    # --- Summary Metrics ---
    date_col = get_date_column(df_logs)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Logs", f"{len(df_logs):,}")
    with col2:
        # Check for user column variants
        user_col = None
        for col in ["user.username", "username", "user"]:
            if col in df_logs.columns:
                user_col = col
                break

        # Count unique users, ignoring NaN/None which can occur if user field is missing
        if user_col is not None and df_logs[user_col].notna().any():
            unique_users = df_logs[user_col].nunique()
            st.metric("Unique Users", f"{unique_users:,}")
        else:
            st.metric("Unique Users", "N/A")
    with col3:
        if "action" in df_logs.columns:
            st.metric("Action Types", f"{df_logs['action'].nunique()}")
        else:
            st.metric("Action Types", "N/A")
    with col4:
        if date_col and not df_logs[date_col].empty:
            date_range = (df_logs[date_col].max() - df_logs[date_col].min()).days
            st.metric("Date Range", f"{date_range} days")
        else:
            st.metric("Date Range", "N/A")

    # --- Interactive Search ---
    st.markdown("### 🔍 Search & Filter")
    search_term = st.text_input("Search in all columns:", key="search_input")

    if search_term:
        mask = df_logs.astype(str).apply(lambda row: row.str.contains(search_term, case=False, na=False).any(), axis=1)
        filtered_df = df_logs[mask]
        st.info(f"Found {len(filtered_df):,} matching rows")
    else:
        filtered_df = df_logs

    st.dataframe(filtered_df, use_container_width=True, height=400)

    # --- Interactive Visualizations ---
    st.markdown("---")
    st.markdown("## 📈 Interactive Visualizations")

    # Row 1: Time series and actions
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        if date_col:
            # Ensure the date column is used correctly and handle potential errors
            try:
                # Use .dt.normalize() to get rid of time components before groupby, ensuring clean daily counts
                df_logs['date_only'] = df_logs[date_col].dt.normalize().dt.date
                daily_counts = df_logs.groupby('date_only').size().reset_index(name='count')
                
                fig1 = px.line(
                    daily_counts,
                    x='date_only',
                    y='count',
                    title=f"📅 {endpoint_name} Over Time (Daily)",
                    markers=True
                )
                fig1.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Number of Logs",
                    hovermode='x unified'
                )
                st.plotly_chart(fig1, use_container_width=True, key="time_chart")
            except Exception as e:
                st.warning(f"Could not generate time series chart: {e}")
        else:
             st.warning("Cannot generate time series chart: Date column not found.")


    with viz_col2:
        if "action" in df_logs.columns:
            action_counts = df_logs['action'].value_counts().head(15).reset_index()
            action_counts.columns = ['action', 'count']
            fig2 = px.bar(
                action_counts,
                x='action',
                y='count',
                title="⚙️ Actions Distribution (Top 15)",
                text='count',
                color='count',
                color_continuous_scale='viridis'
            )
            fig2.update_layout(
                xaxis_title="Action",
                yaxis_title="Count",
                showlegend=False
            )
            fig2.update_traces(textposition='outside')
            st.plotly_chart(fig2, use_container_width=True, key="action_chart")
        else:
             st.warning("Cannot generate action chart: 'action' column not found.")

    # Row 2: User activity and total action breakdown
    viz_col3, viz_col4 = st.columns(2)

    with viz_col3:
        # Use the same 'user_col' logic as in the metrics section
        user_col = None
        for col in ["user.username", "username", "user"]:
            if col in df_logs.columns:
                user_col = col
                break

        if user_col is not None and df_logs[user_col].notna().any():
            # Fill NA values temporarily for counting, then count the non-NA unique users
            user_counts = df_logs[user_col].value_counts().head(15).reset_index()
            user_counts.columns = [user_col, 'count']
            fig3 = px.bar(
                user_counts,
                y=user_col,
                x='count',
                title="👥 Top 15 Active Users",
                text='count',
                orientation='h',
                color='count',
                color_continuous_scale='blues'
            )
            fig3.update_layout(
                yaxis_title="Username",
                xaxis_title="Activity Count",
                showlegend=False,
                height=500,
                yaxis={'categoryorder':'total ascending'}
            )
            st.plotly_chart(fig3, use_container_width=True, key="user_chart")
        else:
             st.warning("Cannot generate user chart: User column not found or no unique users.")

    with viz_col4:
        # Total Action Breakdown (across ALL users)
        if "action" in df_logs.columns:
            total_action_counts = df_logs['action'].value_counts().reset_index()
            total_action_counts.columns = ['action', 'count']

            fig_total_actions = px.pie(
                total_action_counts,
                names='action',
                values='count',
                title="🌐 Actions Breakdown (All Logs)",
            )
            fig_total_actions.update_traces(textposition='inside', textinfo='percent+label')
            fig_total_actions.update_layout(
                 showlegend=True,
                 height=500
            )
            st.plotly_chart(fig_total_actions, use_container_width=True, key="total_action_chart")
        else:
             st.warning("Cannot generate action breakdown: 'action' column not found.")

    # --- Download Section ---
    st.markdown("---")
    st.markdown("## 📥 Export Data")
    st.info("The JSON export contains the **unprocessed, complete nested data** received directly from the API.")

    export_df = create_export_dataframe(df_logs)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename_base = f"kobo_{endpoint_name.lower().replace(' ', '_')}_{timestamp}"

    col1_dl, col2_dl, col3_dl = st.columns(3)

    with col1_dl:
        csv_data = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name=f"{filename_base}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2_dl:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            sheet_name = endpoint_name.replace(" ", "_")[:31] # Max 31 chars
            export_df.to_excel(writer, index=False, sheet_name=sheet_name)

            workbook = writer.book
            worksheet = writer.sheets[sheet_name]

            for i, col in enumerate(export_df.columns):
                # Calculate column width, max 50 characters
                col_width = min(max(export_df[col].astype(str).str.len().max() if not export_df[col].empty else 1, len(col)) + 2, 50)
                worksheet.set_column(i, i, col_width)

            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4CAF50',
                'font_color': 'white'
            })
            worksheet.set_row(0, None, header_format)

        st.download_button(
            label="📊 Download Excel",
            data=output.getvalue(),
            file_name=f"{filename_base}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with col3_dl:
        json_data = json.dumps(raw_logs, indent=2)
        st.download_button(
            label="🧾 Download JSON",
            data=json_data.encode("utf-8"),
            file_name=f"{filename_base}.json",
            mime="application/json",
            use_container_width=True
        )

st.markdown("---")
st.caption(f"🛠️ KoboToolbox Logs Viewer | Server: {server_url if server_url else 'Not configured'}")
