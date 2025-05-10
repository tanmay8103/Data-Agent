import pandas as pd
from typing import Dict, Any

class AnalysisAgent:
    """
    Agent responsible for generating prompts for data analysis.
    """
    
    def prepare_analysis_prompt(self, df: pd.DataFrame) -> str:
        """
        Prepare a comprehensive prompt for analyzing the dataset.
        
        Args:
            df: Input pandas DataFrame
            
        Returns:
            str: Prompt for the LLM
        """
        try:
            # Get basic dataset info
            num_rows = len(df)
            num_cols = len(df.columns)
            columns = df.columns.tolist()
            
            # Get data types info
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            
            # Generate statistical summaries
            # For numerical columns
            num_stats = {}
            if numeric_cols:
                num_stats = df[numeric_cols].describe().to_dict()
            
            # For categorical columns (top 5 categories for each)
            cat_stats = {}
            if categorical_cols:
                for col in categorical_cols:
                    cat_stats[col] = df[col].value_counts().head(5).to_dict()
            
            # Missing values info
            missing_values = {col: int(df[col].isna().sum()) for col in df.columns}
            
            # Create the prompt
            prompt = f"""As a data analysis expert, analyze the following dataset:

Dataset Overview:
- Total Records: {num_rows}
- Total Features: {num_cols}
- Features: {', '.join(columns)}

Data Types:
- Numeric Features: {', '.join(numeric_cols) if numeric_cols else 'None'}
- Categorical Features: {', '.join(categorical_cols) if categorical_cols else 'None'}
- Datetime Features: {', '.join(datetime_cols) if datetime_cols else 'None'}

Numerical Summary:
{num_stats}

Categorical Summary (Top 5 categories):
{cat_stats}

Missing Values:
{missing_values}

Please provide a comprehensive analysis, including:
1. Key patterns and trends in the data
2. Statistical insights from numerical features
3. Distribution patterns in categorical features
4. Notable relationships between features
5. Data quality observations
6. Suggestions for further analysis

Structure your response clearly and focus on actionable insights.
"""
            return prompt
            
        except Exception as e:
            # Fallback to a simpler prompt if there's an error
            print(f"Error preparing analysis prompt: {str(e)}")
            return f"""As a data analysis expert, analyze the following dataset:

Dataset Overview:
- Total Records: {len(df)}
- Total Features: {len(df.columns)}
- Features: {', '.join(df.columns)}

Please provide a comprehensive analysis of this dataset with actionable insights.
"""