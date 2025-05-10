import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

class DataCleaningAgent:
    """
    Simplified data cleaning agent that handles basic data preprocessing
    including missing values, duplicates, outliers, and simple data type issues.
    """
    
    def __init__(self):
        """Initialize the cleaning agent with basic settings"""
        pass
    
    def clean_data(self, df: pd.DataFrame, cleaning_instructions: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main method to clean a dataset with basic operations:
        1. Handle missing values
        2. Remove duplicates
        3. Handle outliers
        4. Standardize text fields
        
        Args:
            df: Input pandas DataFrame
            cleaning_instructions: Optional dictionary with specific cleaning instructions
            
        Returns:
            Tuple containing:
                - Cleaned pandas DataFrame
                - Dictionary with cleaning report details
        """
        # Initialize result dictionary with original data info
        result = {
            "success": True,
            "original_data": {
                "shape": df.shape,
                "columns": df.columns.tolist()
            },
            "cleaning_actions": [],
            "quality_issues": []  # Empty list for compatibility
        }
        
        try:
            # Create a copy to avoid modifying the original
            cleaned_df = df.copy()
            
            # 1. Handle missing values
            cleaned_df, missing_actions = self._handle_missing_values(cleaned_df, cleaning_instructions)
            result["cleaning_actions"].extend(missing_actions)
            
            # 2. Remove duplicates
            original_rows = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            if len(cleaned_df) < original_rows:
                result["cleaning_actions"].append({
                    "column": "all",
                    "action_type": "remove_duplicates",
                    "description": f"Removed {original_rows - len(cleaned_df)} duplicate rows",
                    "justification": "Duplicate rows can skew analysis results"
                })
            
            # 3. Handle outliers in numeric columns
            cleaned_df, outlier_actions = self._handle_outliers(cleaned_df)
            result["cleaning_actions"].extend(outlier_actions)
            
            # 4. Standardize text fields
            cleaned_df, text_actions = self._standardize_text(cleaned_df)
            result["cleaning_actions"].extend(text_actions)
            
            # Add cleaned data info to result
            result["cleaned_data"] = {
                "shape": cleaned_df.shape,
                "columns": cleaned_df.columns.tolist()
            }
            
            return cleaned_df, result
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            result["success"] = False
            result["error"] = str(e)
            result["error_details"] = error_details
            print(f"Error during data cleaning: {str(e)}")
            # Return original data in case of error
            return df, result
    
    def _handle_missing_values(self, df: pd.DataFrame, instructions: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Handle missing values in the dataframe"""
        actions = []
        
        # Fill numeric columns with median
        num_cols = df.select_dtypes(include=['number']).columns
        for col in num_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                df[col] = df[col].fillna(df[col].median())
                actions.append({
                    "column": col,
                    "action_type": "fill_missing",
                    "description": f"Filled {missing_count} missing values with median",
                    "justification": "Median is robust to outliers"
                })
        
        # Fill categorical columns with mode
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                # Use mode if available, otherwise use 'Unknown'
                if df[col].nunique() > 0 and not df[col].mode().empty:
                    mode_value = df[col].mode().iloc[0]
                    df[col] = df[col].fillna(mode_value)
                    actions.append({
                        "column": col,
                        "action_type": "fill_missing",
                        "description": f"Filled {missing_count} missing values with mode: '{mode_value}'",
                        "justification": "Mode represents the most common value"
                    })
                else:
                    df[col] = df[col].fillna('Unknown')
                    actions.append({
                        "column": col,
                        "action_type": "fill_missing",
                        "description": f"Filled {missing_count} missing values with 'Unknown'",
                        "justification": "No mode available, using standard placeholder"
                    })
        
        return df, actions
    
    def _handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Handle outliers in numeric columns using IQR method"""
        actions = []
        
        # Process only numeric columns
        num_cols = df.select_dtypes(include=['number']).columns
        for col in num_cols:
            # Calculate Q1, Q3 and IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier boundaries (using 3*IQR for a conservative approach)
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Count outliers
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            if outliers > 0:
                # Cap outliers to bounds rather than removing them
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                actions.append({
                    "column": col,
                    "action_type": "handle_outliers",
                    "description": f"Capped {outliers} outliers to 3×IQR boundaries",
                    "justification": "Extreme values can distort statistical analysis"
                })
        
        return df, actions
    
    def _standardize_text(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Standardize text fields for consistency"""
        actions = []
        
        # Process only object/string columns
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols:
            # Skip if column has too many unique values (likely not a category)
            if df[col].nunique() > min(100, len(df) // 10):
                continue
                
            # Convert to string if not already
            df[col] = df[col].astype(str)
            
            # Count values with leading/trailing whitespace
            whitespace_count = (df[col].str.len() != df[col].str.strip().str.len()).sum()
            
            if whitespace_count > 0:
                # Strip whitespace
                df[col] = df[col].str.strip()
                actions.append({
                    "column": col,
                    "action_type": "standardize_text",
                    "description": f"Stripped whitespace from {whitespace_count} values",
                    "justification": "Remove inconsistent whitespace for better matching"
                })
            
            # Standardize case for categorical fields (if not too many unique values)
            if df[col].nunique() < min(50, len(df) // 20):
                original_values = df[col].nunique()
                
                # Convert to title case
                df[col] = df[col].str.title()
                
                new_values = df[col].nunique()
                if new_values < original_values:
                    actions.append({
                        "column": col,
                        "action_type": "standardize_text",
                        "description": f"Standardized text case, reducing unique values from {original_values} to {new_values}",
                        "justification": "Standardize case for better category matching"
                    })
        
        return df, actions