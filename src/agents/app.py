# app.py (Complete Updated Code - Indentation Corrected)
import streamlit as st
import pandas as pd
import os
import json
import plotly.graph_objects as go
from dotenv import load_dotenv
# Assuming these files are in the same directory or correctly imported
from workflow import create_analysis_workflow
from data_loader import load_data_from_db # Import the loader

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    st.error("⚠️ OPENAI_API_KEY not found in .env file. Please add it to proceed.")
    st.stop()

# Helper function to display visualizations with error handling and debug info
def display_visualizations(viz_data, section_title, debug_mode=False, key_prefix="viz"):
    """
    Helper function to display visualizations with error handling and debug info

    Args:
        viz_data: Visualization data from agent
        section_title: Title for the visualization section
        debug_mode: Whether to show debug information
        key_prefix: Prefix for unique Streamlit keys to avoid duplicate ID errors
    """
    if not viz_data:
        st.write(f"### {section_title}")
        st.info("No visualization data available")
        return

    # Display debug info if enabled
    if debug_mode:
        with st.expander("🔍 Visualization Debug Info", expanded=False):
            if isinstance(viz_data, dict) and "debug_info" in viz_data:
                st.json(viz_data["debug_info"])

            st.write(f"Success: {viz_data.get('success', False)}")
            st.write(f"Fallback used: {viz_data.get('is_fallback', False)}")
            st.write(f"Number of plots: {len(viz_data.get('plots', []))}")

    # Check if visualizations were generated successfully
    if isinstance(viz_data, dict) and viz_data.get("success"):
        st.write(f"### {section_title}")

        # If using fallback visualizations, show a notice
        if viz_data.get("is_fallback", False):
            st.info("⚠️ Using fallback visualizations due to issues with AI-generated visualizations")

        # Get the plots list
        plots = viz_data.get("plots", [])
        if not isinstance(plots, list):
            st.error("Invalid visualization data format")
            return

        # Create tabs for multiple visualizations
        if len(plots) > 1:
            tab_titles = [plot.get("title", f"Visualization {i+1}") for i, plot in enumerate(plots)]
            tabs = st.tabs(tab_titles)

            for i, (tab, plot_data) in enumerate(zip(tabs, plots)):
                with tab:
                    try:
                        st.subheader(plot_data.get("title", f"Visualization {i+1}"))

                        # Convert the JSON string back to a Plotly figure
                        if isinstance(plot_data.get("figure"), str):
                            fig_dict = json.loads(plot_data["figure"])
                            fig = go.Figure(fig_dict)

                            # Display the figure with a unique key
                            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_tab_{i}_{plot_data.get('title', '').replace(' ', '_')}")

                            # Add insightful description
                            if plot_data.get("insight"):
                                st.write(plot_data["insight"])
                    except Exception as e:
                        st.error(f"Error displaying visualization: {str(e)}")
                        if debug_mode:
                            import traceback
                            st.code(traceback.format_exc())
        elif len(plots) == 1:
            # Single visualization
            try:
                plot_data = plots[0]
                st.subheader(plot_data.get("title", "Visualization"))

                # Convert the JSON string back to a Plotly figure
                if isinstance(plot_data.get("figure"), str):
                    fig_dict = json.loads(plot_data["figure"])
                    fig = go.Figure(fig_dict)

                    # Display the figure with a unique key
                    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_single_{plot_data.get('title', '').replace(' ', '_')}")

                    # Add insightful description
                    if plot_data.get("insight"):
                        st.write(plot_data["insight"])
            except Exception as e:
                st.error(f"Error displaying visualization: {str(e)}")
                if debug_mode:
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.info("No visualizations were generated.")
    else:
        st.write(f"### {section_title}")

        # Show error message if visualizations failed
        if isinstance(viz_data, dict) and not viz_data.get("success"):
            if "error" in viz_data:
                st.error(f"❌ Visualization failed: {viz_data['error']}")
            else:
                st.error("❌ Failed to generate visualizations")

            # Create fallback visualization on the fly if necessary
            st.info("Creating basic visualizations...")

            # Create a simple data summary
            st.write("#### Data Summary")
            # Use the dataframe stored in session state (either original or cleaned if cleaning didn't crash)
            if "current_df_display" in st.session_state and st.session_state["current_df_display"] is not None:
                df_display = st.session_state["current_df_display"]

                # Display numeric columns summary
                numeric_cols = df_display.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    st.write("Summary of numeric columns:")
                    st.dataframe(df_display[numeric_cols].describe(include='all'))

                    # Create a simple bar chart/histogram for the first few numeric columns
                    st.write("Sample Distributions:")
                    for i, col in enumerate(numeric_cols[:3]): # Limit to first 3 for brevity
                         try:
                             st.write(f"Distribution of {col}")
                             # Use Streamlit's built-in bar chart for a quick fallback viz
                             # Need to ensure col is numeric and drop NaNs for st.bar_chart
                             chart_data = df_display[col].dropna()
                             if not chart_data.empty:
                                  # For distributions, a histogram-like bar chart of value counts is often useful
                                  # But a simple bar chart of the series is also quick
                                  # Let's just use st.histgram for numeric series
                                  fig_fallback_hist = go.Figure(data=go.Histogram(x=chart_data))
                                  fig_fallback_hist.update_layout(title=f"Distribution of {col}", height=300)
                                  st.plotly_chart(fig_fallback_hist, use_container_width=True, key=f"{key_prefix}_fallback_hist_{col}")

                             else:
                                  st.write(f"Cannot display chart for {col} as it contains no valid data.")
                         except Exception as e:
                              st.write(f"Error creating fallback chart for {col}: {e}")

                # Display categorical column counts
                cat_cols = df_display.select_dtypes(include=['object', 'category']).columns
                if len(cat_cols) > 0:
                     st.write("Summary of categorical columns (Top 10):")
                     for i, col in enumerate(cat_cols[:3]): # Limit to first 3
                         if df_display[col].count() > 0:
                             st.write(f"Counts for {col}")
                             # Display top value counts
                             top_values = df_display[col].value_counts().head(10)
                             st.dataframe(top_values) # Using dataframe for counts is simple

                     else:
                         st.info("No data available to summarize.")
        else:
            st.info("No visualization data available")


# Define a cached function for CSV loading
@st.cache_data(show_spinner="Loading CSV data...")
def load_csv_data(uploaded_file):
    """Loads data from an uploaded CSV file."""
    try:
        # Use uploaded_file.getvalue() for pandas read_csv
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"⚠️ Error loading CSV file: {str(e)}")
        return None


# Define a function for DB loading (can be cached if query/config don't change often)
# Caching might be complex if the underlying DB data changes frequently
# @st.cache_data(show_spinner="Loading data from database...") # Add caching if appropriate
def load_db_data(db_config, query):
    """Loads data from a database using the data_loader."""
    # The actual loading logic is in data_loader.py
    try:
        return load_data_from_db(db_config, query)
    except Exception as e:
        # Error message is already shown by load_data_from_db
        return None


def main():
    st.set_page_config(layout="wide") # Use wide layout
    st.title("📊 AI-Powered Data Analysis")
    st.write("Analyze data with AI, including cleaning and visualization, from CSV or Database sources.")

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": "Hello! Upload a CSV or connect to a database to begin. You can also chat with me here to influence the cleaning process (e.g., 'impute numeric with mean')."}]

    # Initialize data source state
    if "data_source" not in st.session_state:
        st.session_state.data_source = "CSV Upload"

    # Initialize DataFrame state (the original loaded data)
    if "df" not in st.session_state:
        st.session_state.df = None
    # State to hold the DataFrame currently being displayed/used for preview (can be original or cleaned)
    if "current_df_display" not in st.session_state:
        st.session_state.current_df_display = None

    # Initialize analysis results state
    if "original_analysis_result" not in st.session_state:
         st.session_state.original_analysis_result = ""
    if "original_visualization_data" not in st.session_state:
         st.session_state.original_visualization_data = None
    if "cleaning_results" not in st.session_state:
         st.session_state.cleaning_results = None
    if "cleaned_analysis_result" not in st.session_state:
         st.session_state.cleaned_analysis_result = ""
    if "cleaned_visualization_data" not in st.session_state:
         st.session_state.cleaned_visualization_data = None
    if "debug_logs" not in st.session_state:
         st.session_state.debug_logs = []
    # State to hold the final cleaned dataframe from the workflow
    if "cleaned_df" not in st.session_state:
         st.session_state.cleaned_df = None


    # Sidebar Configuration
    with st.sidebar:
        st.title("Configuration")
        debug_mode = st.checkbox("Enable Debug Mode", value=True)

        st.title("Data Source")
        # Add a key to the radio button
        data_source = st.radio("Select Data Source", ["CSV Upload", "Database"], key="data_source_radio")
        # Update session state based on radio button change
        if data_source != st.session_state.data_source:
             st.session_state.data_source = data_source
             # Clear previous data and results when changing source type
             st.session_state.df = None
             st.session_state.current_df_display = None
             st.session_state.original_analysis_result = ""
             st.session_state.original_visualization_data = None
             st.session_state.cleaning_results = None
             st.session_state.cleaned_analysis_result = ""
             st.session_state.cleaned_visualization_data = None
             st.session_state.debug_logs = []
             st.session_state.cleaned_df = None


        # --- Data Loading UI ---
        df_loaded_in_this_run = None # Flag to track if data was newly loaded in this run

        if st.session_state.data_source == "CSV Upload":
            uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
            if uploaded_file is not None: # Check if a file was actually uploaded
                # Use the cached loader
                df_loaded = load_csv_data(uploaded_file)
                if df_loaded is not None:
                     # Only update state if a new file is uploaded or the content changes
                     # Streamlit's file_uploader/cache handles this implicitly
                     # Check if loaded df is different from the one in state to avoid unnecessary reruns/messages
                     if st.session_state.df is None or not st.session_state.df.equals(df_loaded):
                          st.session_state.df = df_loaded
                          st.session_state.current_df_display = df_loaded.copy()
                          df_loaded_in_this_run = True # Mark data as loaded
                          st.session_state.chat_history.append({"role": "assistant", "content": f"CSV loaded successfully! Dataset has {len(df_loaded)} rows and {len(df_loaded.columns)} columns. What cleaning instruction would you like to give? Or click 'Analyze Data' to proceed with default cleaning."})

        elif st.session_state.data_source == "Database":
            st.subheader("Database Connection Details")
            if "db_config" not in st.session_state:
                st.session_state.db_config = {
                    "type": "postgresql", # Default, will be changed by selectbox
                    "host": "",
                    "port": "",
                    "database": "",
                    "user": "",
                    "password": ""
                }

            # Database Type Selector - Add sqlite option
            # Use a key for selectbox
            db_type_selected = st.selectbox(
                "Database Type",
                ["postgresql", "mysql", "sqlserver", "sqlite"],
                key="db_type_selector" # Changed key name for clarity
            )
            st.session_state.db_config["type"] = db_type_selected # Update config type


            # --- Conditional fields based on DB Type ---
            is_sqlite = st.session_state.db_config["type"] == "sqlite"

            # Non-SQLite fields
            if not is_sqlite:
                 # These lines should be at the same indentation as the st.selectbox above
                 st.session_state.db_config["host"] = st.text_input("Host", value=st.session_state.db_config.get("host", ""), key="db_host")
                 st.session_state.db_config["port"] = st.text_input("Port", value=st.session_state.db_config.get("port", ""), key="db_port")
                 st.session_state.db_config["user"] = st.text_input("User", value=st.session_state.db_config.get("user", ""), key="db_user")
                 st.session_state.db_config["password"] = st.text_input("Password", type="password", value=st.session_state.db_config.get("password", ""), key="db_password")
                 st.session_state.db_config["database"] = st.text_input("Database Name", value=st.session_state.db_config.get("database", ""), key="db_name")

            else:
                 # SQLite field: Database File Path
                 # These lines should be at the same indentation as the st.selectbox above
                 st.session_state.db_config["database"] = st.text_input("Database File Path", value=st.session_state.db_config.get("database", ""), key="db_filepath")
                 # Explicitly clear/reset non-sqlite fields when switching TO sqlite
                 st.session_state.db_config["host"] = ""
                 st.session_state.db_config["port"] = ""
                 st.session_state.db_config["user"] = ""
                 st.session_state.db_config["password"] = ""
                 # The 'database' key now holds the file path


            st.subheader("Query or Table")
            if "db_query" not in st.session_state:
                st.session_state.db_query = ""

            # Different UI for SQLite vs other databases
            if is_sqlite:
                # For SQLite, show available tables
                try:
                    import sqlite3
                    conn = sqlite3.connect(st.session_state.db_config["database"])
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    conn.close()
                    
                    if tables:
                        table_names = [table[0] for table in tables]
                        selected_table = st.selectbox("Select Table", table_names)
                        st.session_state.db_query = f"SELECT * FROM {selected_table}"
                        
                        # Show preview of selected table
                        if selected_table:
                            conn = sqlite3.connect(st.session_state.db_config["database"])
                            preview_df = pd.read_sql_query(f"SELECT * FROM {selected_table} LIMIT 5", conn)
                            conn.close()
                            
                            st.write("### Table Preview")
                            st.dataframe(preview_df)
                            st.write(f"Table has {len(preview_df)} preview rows")
                    else:
                        st.warning("No tables found in the SQLite database.")
                except Exception as e:
                    st.error(f"Error connecting to SQLite database: {str(e)}")
            else:
                # For other databases, keep the query input
                st.session_state.db_query = st.text_area("SQL Query or Table Name", value=st.session_state.db_query, key="db_query_input", height=100)

            # Button to load data from DB *before* running analysis workflow
            # Add a key to the button
            if st.button("Load Data from Database", key="load_db_button"):
                # --- Adjust Validation for SQLite ---
                is_sqlite = st.session_state.db_config["type"] == "sqlite"
                is_valid_config = False

                if is_sqlite:
                    # For SQLite, only the file path (in 'database' key) is needed
                    if st.session_state.db_config.get("database"):
                         is_valid_config = True
                    else:
                         st.warning("Please provide the Database File Path for SQLite.")
                else:
                    # For other DBs, check host, database, user, and query/table
                    if st.session_state.db_config.get("host") and st.session_state.db_config.get("database") and st.session_state.db_config.get("user") and st.session_state.db_query:
                         is_valid_config = True
                    else:
                         st.warning(f"Please fill in all required database details (Host, Database, User) and provide a Query or Table Name for {st.session_state.db_config['type']}.")

                # --- End Validation Adjustment ---

                if is_valid_config:
                    # Use the API key from environment variables
                    if not api_key:
                        st.error("⚠️ OPENAI_API_KEY not found in .env file. Please add it to proceed.")
                    else:
                        # Clear previous data/results before loading new DB data
                        st.session_state.df = None
                        st.session_state.current_df_display = None
                        st.session_state.original_analysis_result = ""
                        st.session_state.original_visualization_data = None
                        st.session_state.cleaning_results = None
                        st.session_state.cleaned_analysis_result = ""
                        st.session_state.cleaned_visualization_data = None
                        st.session_state.debug_logs = []
                        st.session_state.cleaned_df = None

                        with st.spinner(f"Loading data from {st.session_state.db_config['type']}..."):
                            df_loaded = load_db_data(st.session_state.db_config, st.session_state.db_query)
                            if df_loaded is not None:
                                 st.session_state.df = df_loaded # Store loaded df in state
                                 st.session_state.current_df_display = df_loaded.copy() # Store for preview
                                 df_loaded_in_this_run = True # Mark data as loaded
                                 st.success("Data loaded successfully!")
                                 st.session_state.chat_history.append({"role": "assistant", "content": f"Data loaded successfully from the database. Dataset has {len(df_loaded)} rows and {len(df_loaded.columns)} columns. What cleaning instruction would you like to give? Or click 'Analyze Data' to proceed with default cleaning."})
                            # If loading failed, the error is shown by load_db_data, state remains None


        # --- Display loaded data preview if available in state ---
        # Use st.session_state.current_df_display for preview as it holds the current data for display
        if st.session_state.current_df_display is not None:
            st.write("### Data Preview")
            st.dataframe(st.session_state.current_df_display.head(10)) # Show first 10 rows
            st.write(f"Dataset has {len(st.session_state.current_df_display)} rows and {len(st.session_state.current_df_display.columns)} columns")

            # Display column info if debug mode is on
            if debug_mode:
                st.write("### Column Information")
                # Use st.session_state.current_df_display for column info
                df_display_info = st.session_state.current_df_display
                col_info = {
                    "Column": df_display_info.columns.tolist(),
                    "Type": [str(df_display_info[col].dtype) for col in df_display_info.columns],
                    "Non-Null Count": [df_display_info[col].count() for col in df_display_info.columns],
                    "Null Count": [df_display_info[col].isna().sum() for col in df_display_info.columns],
                    "Unique Values": [df_display_info[col].nunique() for col in df_display_info.columns]
                }
                st.dataframe(pd.DataFrame(col_info).set_index("Column"))


        # --- Chat Interface ---
        st.write("### Chat with the AI")
        # Display chat messages from session state
        # Add a unique key for the chat message container itself
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                # Add a unique key for each message container inside the loop
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        # Chat input - only enable if data is loaded
        if st.session_state.df is not None: # Check the original loaded df state
            prompt = st.chat_input("Enter cleaning instructions (e.g., 'impute numeric with mean') or ask questions about the data...", key="chat_input")
        else:
            prompt = st.chat_input("Load data first to chat about cleaning...", key="chat_input", disabled=True)


        # If a new prompt is entered, add it to the chat history
        # Streamlit handles prompt submission on Enter
        if prompt:
            # Append user message to chat history immediately
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            # Rerun the app to display the new message and potentially update based on it
            st.rerun() # st.rerun is needed to immediately show the message and trigger workflow logic if analyze is clicked next


        # --- Analyze Data Button ---
        # Only show the analyze button if original loaded data is present (df state)
        if st.session_state.df is not None:
            # Add a key to the analyze button
            if st.button("Analyze Data", key="analyze_button"):
                 # Double-check df validity just before running workflow
                 if st.session_state.df is None or len(st.session_state.df) == 0 or len(st.session_state.df.columns) == 0:
                     st.error("Cannot analyze empty or invalid data.")
                 else:
                    # Clear previous analysis results before running a new analysis
                    st.session_state.original_analysis_result = ""
                    st.session_state.original_visualization_data = None
                    st.session_state.cleaning_results = None
                    st.session_state.cleaned_analysis_result = ""
                    st.session_state.cleaned_visualization_data = None
                    st.session_state.debug_logs = []
                    st.session_state.cleaned_df = None # Clear previous cleaned data


                    with st.spinner("Running analysis workflow..."):
                        try:
                            # Create and run the analysis workflow
                            # Pass the original loaded dataframe (df state) and the current chat history from state
                            analysis_runner = create_analysis_workflow(
                                df=st.session_state.df, # Use the original loaded df as input to workflow
                                chat_history=st.session_state.chat_history, # Pass chat history to workflow
                                debug_mode=debug_mode,
                                api_key=api_key  # Use the API key from environment
                            )

                            # Run the workflow
                            # The workflow returns the final state
                            final_workflow_state = analysis_runner()

                            # Update session state with results and cleaned df for display from the final state
                            st.session_state.original_analysis_result = final_workflow_state.get("original_analysis", {}).get("result", "")
                            st.session_state.original_visualization_data = final_workflow_state.get("original_analysis", {}).get("visualizations")
                            st.session_state.cleaning_results = final_workflow_state.get("cleaning")
                            st.session_state.cleaned_analysis_result = final_workflow_state.get("cleaned_analysis", {}).get("result", "")
                            st.session_state.cleaned_visualization_data = final_workflow_state.get("cleaned_analysis", {}).get("visualizations")
                            st.session_state.debug_logs = final_workflow_state.get("debug_logs", [])

                            # Store the final cleaned dataframe returned by the workflow
                            st.session_state.cleaned_df = final_workflow_state.get("cleaned_df")

                            # Update the DataFrame for preview to the cleaned one if available and valid
                            if st.session_state.cleaned_df is not None and not st.session_state.cleaned_df.empty:
                                st.session_state.current_df_display = st.session_state.cleaned_df.copy() # Use cleaned for final preview
                                # Add a system message to chat indicating workflow completion
                                cleaning_summary_chat = "Analysis complete."
                                if final_workflow_state.get("cleaning", {}).get("success", False):
                                     orig_shape = final_workflow_state['cleaning']['original_data']['shape']
                                     clean_shape = final_workflow_state['cleaning']['cleaned_data']['shape']
                                     cleaning_summary_chat += f" Data has been cleaned (shape {orig_shape} -> {clean_shape})"
                                     # Get instructions applied from the cleaning result within the final state
                                     cleaning_instructions_applied = final_workflow_state.get("cleaning_results", {}).get("cleaning_instructions", {}) # Get from workflow results if stored
                                     if cleaning_instructions_applied:
                                          cleaning_summary_chat += f" with instructions: {cleaning_instructions_applied}"

                                cleaning_summary_chat += ". Here are the results and visualizations."
                                st.session_state.chat_history.append({"role": "assistant", "content": cleaning_summary_chat})

                            else:
                                # If cleaning failed or returned empty, use the original for display preview
                                # st.session_state.current_df_display is already the original if cleaning failed in workflow
                                st.session_state.chat_history.append({"role": "assistant", "content": "Analysis complete. Cleaning may have failed or returned no data, displaying analysis of original data and visualizations. Check debug logs for details."})

                            st.success("Analysis workflow finished.")
                            # Trigger rerun to display updated state, including new chat message
                            st.rerun()

                        except Exception as e:
                            st.error(f"An error occurred during the analysis workflow: {str(e)}")
                            import traceback
                            error_trace = traceback.format_exc()
                            if debug_mode:
                                st.code(error_trace)
                            st.session_state.chat_history.append({"role": "assistant", "content": f"An error occurred during analysis: {str(e)}. Please check the debug logs if enabled."})
                            # Ensure display data is not None even if workflow failed, fallback to original
                            if st.session_state.df is not None:
                                 st.session_state.current_df_display = st.session_state.df.copy()
                            st.rerun() # Rerun to show error and new chat message


    # --- Display Results Sections (based on session state) ---
    # Display debug logs if in debug mode and available
    if debug_mode and "debug_logs" in st.session_state and st.session_state.debug_logs:
         with st.expander("🔍 Debug Logs", expanded=False):
             for i, log in enumerate(st.session_state.debug_logs):
                 st.text(f"{i+1}. {log}")

    # Display analysis and visualizations if results are in session state
    if "original_analysis_result" in st.session_state and st.session_state.original_analysis_result:
        st.write("## Original Data Analysis")
        st.write(st.session_state.original_analysis_result)

    if "original_visualization_data" in st.session_state and st.session_state.original_visualization_data:
        display_visualizations(
            st.session_state.original_visualization_data,
            "Original Data Visualizations",
            debug_mode,
            "original"
        )

    if "cleaning_results" in st.session_state and st.session_state.cleaning_results:
        cleaning_results = st.session_state.cleaning_results
        st.write("## Data Cleaning Results")

        # Safely get the counts with type checking
        quality_issues = cleaning_results.get('quality_issues', [])
        cleaning_actions = cleaning_results.get('cleaning_actions', [])

        # Make sure these are lists before counting
        quality_issues_count = len(quality_issues) if isinstance(quality_issues, list) else 0
        cleaning_actions_count = len(cleaning_actions) if isinstance(cleaning_actions, list) else 0

        st.write(f"- Quality issues identified: {quality_issues_count}")
        st.write(f"- Cleaning actions performed: {cleaning_actions_count}")

        # Display cleaning instructions applied, if any were detected by the workflow
        # Get the instructions from the cleaning_results part of the state
        cleaning_instructions_applied = st.session_state.get("cleaning_results", {}).get("cleaning_instructions")
        if cleaning_instructions_applied:
             st.write(f"**Cleaning Instructions Applied:** {cleaning_instructions_applied}")


        # Display top cleaning actions with type checking
        if isinstance(cleaning_actions, list) and len(cleaning_actions) > 0:
            st.write("### Top Cleaning Actions:")
            count = 0
            for action in cleaning_actions:
                if isinstance(action, dict):
                     st.write(f"- {action.get('column', 'dataset')}: {action.get('description', 'N/A')}")
                elif isinstance(action, str):
                    st.write(f"- {action}")
                count += 1
                if count >= 5:
                    break

        if 'original_data' in cleaning_results and 'cleaned_data' in cleaning_results:
            orig_shape = cleaning_results['original_data'].get('shape', (0, 0))
            clean_shape = cleaning_results['cleaned_data'].get('shape', (0, 0))

            if orig_shape != clean_shape:
                st.write(f"Data shape changed from {orig_shape} to {clean_shape}")

        if debug_mode and cleaning_results.get('warning_flags'):
            st.warning("Cleaning Warnings:")
            for warning in cleaning_results.get('warning_flags', []):
                st.write(f"- {warning}")

    if "cleaned_analysis_result" in st.session_state and st.session_state.cleaned_analysis_result:
        st.write("## Cleaned Data Analysis")
        st.write(st.session_state.cleaned_analysis_result)

    if "cleaned_visualization_data" in st.session_state and st.session_state.cleaned_visualization_data:
        display_visualizations(
            st.session_state.cleaned_visualization_data,
            "Cleaned Data Visualizations",
            debug_mode,
            "cleaned"
        )

    # Show preview of final cleaned data if available
    # This uses the 'cleaned_df' state directly, which is set by the workflow
    if "cleaned_df" in st.session_state and st.session_state.cleaned_df is not None and len(st.session_state.cleaned_df) > 0:
         st.write("## Final Cleaned Data Preview")
         st.dataframe(st.session_state.cleaned_df.head(10)) # Show first 10 rows
         st.write(f"Final cleaned dataset has {len(st.session_state.cleaned_df)} rows and {len(st.session_state.cleaned_df.columns)} columns")
    elif "cleaning_results" in st.session_state and st.session_state.cleaning_results:
         # If cleaning results exist but cleaned_df is None/empty, it implies cleaning failed
         st.warning("Cleaned data could not be displayed after processing.")


if __name__ == "__main__":
    main()