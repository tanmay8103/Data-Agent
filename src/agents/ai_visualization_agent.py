import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import json
import io
from typing import Dict, Any, List
import os
from openai import OpenAI
import traceback
from datetime import datetime

class PandasJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling Pandas objects"""
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class VisualizationAgent:
    """
    Enhanced visualization agent with a wider variety of plot types and improved
    recommendations based on data characteristics.
    """
    
    def __init__(self, debug=False, api_key=None):
        """Initialize the visualization agent with API client"""
        self.api_key = api_key
        self.debug = debug
        if not self.api_key:
            raise ValueError("No API Key found. Please set OPENAI_API_KEY in .env file.")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def create_visualization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create a diverse set of visualizations based on dataset analysis with enhanced error handling.
        """
        viz_data = {"success": False, "plots": [], "debug_info": {}}
        
        # Early validation - if dataframe is empty or None, return with error
        if df is None or len(df) == 0 or len(df.columns) == 0:
            viz_data["debug_info"]["error"] = "Empty or invalid dataframe provided"
            return viz_data
            
        try:
            # Add more debugging info about the dataframe
            if self.debug:
                viz_data["debug_info"]["df_shape"] = df.shape
                viz_data["debug_info"]["df_columns"] = df.columns.tolist()
                viz_data["debug_info"]["df_dtypes"] = {col: str(df[col].dtype) for col in df.columns}
                viz_data["debug_info"]["df_null_counts"] = {col: int(df[col].isna().sum()) for col in df.columns}
            
            # Get column info
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Try to detect datetime columns - both explicit datetime and string columns that might contain dates
            date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            
            # Check for string columns that might be dates
            for col in cat_cols:
                if col not in date_cols:
                    # Sample some values and see if they could be parsed as dates
                    sample = df[col].dropna().head(5)
                    try_date_count = 0
                    
                    for val in sample:
                        try:
                            if pd.to_datetime(val, errors='coerce') is not pd.NaT:
                                try_date_count += 1
                        except:
                            pass
                    
                    # If 3+ of 5 values can be parsed as dates, consider it a date column
                    if try_date_count >= 3:
                        date_cols.append(col)
            
            # Print debug info
            if self.debug:
                print(f"Numeric columns: {num_cols}")
                print(f"Categorical columns: {cat_cols}")
                print(f"Date/time columns: {date_cols}")
            
            # Convert sample data to a format that can be JSON serialized
            sample_data = df.head(5).copy()
            for col in sample_data.columns:
                if pd.api.types.is_datetime64_any_dtype(sample_data[col]):
                    sample_data[col] = sample_data[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Dataset info for LLM
            dataset_info = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "numeric_columns": num_cols,
                "categorical_columns": cat_cols,
                "datetime_columns": date_cols,
                "sample_data": sample_data.to_dict(orient='records')
            }
            
            # Create enhanced prompt for the LLM
            prompt = f"""As a data visualization expert, analyze this dataset and recommend diverse visualizations.

Dataset Info:
```json
{json.dumps(dataset_info, indent=2, cls=PandasJSONEncoder)}
```

Numeric columns: {', '.join(num_cols) if num_cols else 'None'}
Categorical columns: {', '.join(cat_cols) if cat_cols else 'None'}
Potential date/time columns: {', '.join(date_cols) if date_cols else 'None'}

For each numeric column, here are basic statistics:
"""
            # Add basic stats for a few columns to help the LLM understand the data better
            for col in num_cols[:3]:  # Limit to first 3 numeric columns
                try:
                    stats = df[col].describe().to_dict()
                    prompt += f"\n{col}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}, std={stats['std']:.2f}"
                except Exception as e:
                    if self.debug:
                        print(f"Error getting stats for column {col}: {str(e)}")
                    prompt += f"\n{col}: stats unavailable"
            
            prompt += """

RECOMMEND 5-8 DIVERSE VISUALIZATIONS with these details:
1. Which columns to visualize
2. What chart type to use (be specific and diverse - consider histograms, scatter plots, bar charts, line charts, pie charts, heatmaps, area charts, bubble charts, violin plots, radar charts, etc.)
3. Why this visualization would be insightful

For time series or date columns, consider line charts, area charts, and calendar heatmaps.
For correlations, consider heatmaps, scatter plot matrices, or bubble charts.
For distributions, consider histograms, violin plots, or box plots.
For compositions, consider pie charts, treemaps, or stacked area charts.
For comparisons, consider bar charts, radar charts, or parallel coordinates.

DO NOT provide any code. Just describe the visualizations you recommend.
"""
            
            # Call the OpenAI API with error handling
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",  
                    messages=[
                        {"role": "system", "content": "You are an expert data visualization specialist who provides creative, diverse visualization recommendations based on data characteristics."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                
                # Get LLM recommendations
                llm_response = response.choices[0].message.content
                
                if self.debug:
                    viz_data["debug_info"]["llm_response"] = llm_response
                
                # Parse LLM response and create visualizations
                plots = self._create_visualizations_from_recommendations(df, llm_response)
                
                if not plots:
                    # If no visualizations could be created from recommendations, use standard visualizations
                    if self.debug:
                        viz_data["debug_info"]["fallback_reason"] = "No plots created from LLM recommendations"
                    return self._create_standard_visualizations(df)
                
                # Add plots to result
                viz_data["plots"] = plots
                viz_data["success"] = True
                
                return viz_data
                
            except Exception as e:
                if self.debug:
                    viz_data["debug_info"]["api_exception"] = str(e)
                    viz_data["debug_info"]["api_traceback"] = traceback.format_exc()
                # If API call throws an exception, fall back to standard visualizations
                return self._create_standard_visualizations(df)
                
        except Exception as e:
            if self.debug:
                print(f"Error in create_visualization: {str(e)}")
                print(traceback.format_exc())
                viz_data["debug_info"]["error"] = str(e)
                viz_data["debug_info"]["traceback"] = traceback.format_exc()
            
            # If anything goes wrong, create standard visualizations
            return self._create_standard_visualizations(df)
    
    def _create_visualizations_from_recommendations(self, df: pd.DataFrame, recommendations: str) -> List[Dict[str, Any]]:
        """
        Parse LLM recommendations and create an enhanced set of visualizations.
        """
        plots = []
        
        try:
            # Look for visualization cues in the recommendations
            lines = recommendations.split('\n')
            current_section = None
            viz_descriptions = []
            current_viz = {}
            
            # Parse the LLM response in a simple way
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Look for numbered recommendations
                if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) and len(line) > 3:
                    # Save previous viz if exists
                    if current_viz:
                        viz_descriptions.append(current_viz)
                        current_viz = {}
                    
                    current_viz = {"description": line}
                elif current_viz:
                    # Add to current viz description
                    current_viz["description"] = current_viz.get("description", "") + " " + line
            
            # Add the last viz
            if current_viz:
                viz_descriptions.append(current_viz)
            
            # If we couldn't parse any visualization descriptions, return empty list
            if not viz_descriptions:
                if self.debug:
                    print("No visualization descriptions found in LLM response")
                return []
            
            # Get column info
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            
            # Try to detect string columns that might be dates
            potential_date_cols = []
            for col in cat_cols:
                try:
                    # Try to convert to datetime with common formats
                    test = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
                    if test.notna().mean() < 0.8:
                        test = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')
                    if test.notna().mean() < 0.8:
                        test = pd.to_datetime(df[col], format='%m/%d/%Y', errors='coerce')
                    if test.notna().mean() < 0.8:
                        test = pd.to_datetime(df[col], format='%Y/%m/%d', errors='coerce')
                    
                    # If at least 80% of values can be converted, consider it a date column
                    if test.notna().mean() >= 0.8:
                        potential_date_cols.append(col)
                        # Also convert it for use in visualizations
                        df[col + '_date'] = test
                        date_cols.append(col + '_date')
                except:
                    pass
            
            # Create visualizations based on description patterns
            for idx, viz in enumerate(viz_descriptions):
                try:
                    desc = viz.get("description", "").lower()
                    
                    # Skip if description is too short or empty
                    if len(desc) < 10:
                        continue
                    
                    # Look for mentions of histogram
                    if any(term in desc for term in ['histogram', 'distribution']):
                        self._add_histogram_plot(df, num_cols, desc, idx, plots, viz)
                    
                    # Look for bar chart
                    elif any(term in desc for term in ['bar chart', 'bar graph', 'counts', 'frequency']):
                        self._add_bar_plot(df, cat_cols, desc, idx, plots, viz)
                    
                    # Look for pie chart
                    elif any(term in desc for term in ['pie chart', 'composition', 'breakdown', 'proportion']):
                        self._add_pie_plot(df, cat_cols, desc, idx, plots, viz)
                    
                    # Look for scatter plot
                    elif any(term in desc for term in ['scatter', 'relationship', 'correlation']):
                        self._add_scatter_plot(df, num_cols, cat_cols, desc, idx, plots, viz)
                    
                    # Look for box plot
                    elif any(term in desc for term in ['box plot', 'boxplot', 'box-and-whisker']):
                        self._add_box_plot(df, num_cols, cat_cols, desc, idx, plots, viz)
                    
                    # Look for line chart
                    elif any(term in desc for term in ['line chart', 'line graph', 'trend', 'time series']):
                        self._add_line_plot(df, num_cols, date_cols, cat_cols, desc, idx, plots, viz)
                    
                    # Look for area chart
                    elif any(term in desc for term in ['area chart', 'area graph', 'cumulative', 'stacked area']):
                        self._add_area_plot(df, num_cols, date_cols, cat_cols, desc, idx, plots, viz)
                    
                    # Look for heatmap
                    elif any(term in desc for term in ['heatmap', 'heat map', 'correlation matrix']):
                        self._add_heatmap_plot(df, num_cols, desc, idx, plots, viz)
                    
                    # Look for bubble chart
                    elif any(term in desc for term in ['bubble chart', 'bubble plot']):
                        self._add_bubble_plot(df, num_cols, cat_cols, desc, idx, plots, viz)
                    
                    # Look for violin plot
                    elif any(term in desc for term in ['violin plot', 'violin']):
                        self._add_violin_plot(df, num_cols, cat_cols, desc, idx, plots, viz)
                    
                    # Look for radar/spider chart
                    elif any(term in desc for term in ['radar chart', 'radar plot', 'spider chart', 'spider plot']):
                        self._add_radar_plot(df, num_cols, cat_cols, desc, idx, plots, viz)
                    
                    # Look for treemap
                    elif any(term in desc for term in ['treemap', 'tree map']):
                        self._add_treemap_plot(df, cat_cols, num_cols, desc, idx, plots, viz)
                    
                except Exception as e:
                    if self.debug:
                        print(f"Error creating visualization from recommendation: {str(e)}")
                    continue
            
            return plots
            
        except Exception as e:
            if self.debug:
                print(f"Error in _create_visualizations_from_recommendations: {str(e)}")
                print(traceback.format_exc())
            return []
    
    def _add_histogram_plot(self, df, num_cols, desc, idx, plots, viz):
        """Helper method to add a histogram plot."""
        for col in num_cols:
            if col.lower() in desc.lower():
                try:
                    fig = px.histogram(
                        df, 
                        x=col, 
                        title=f"Distribution of {col}",
                        template='plotly_white',
                        opacity=0.8,
                        color_discrete_sequence=['#636EFA']
                    )
                    fig.update_layout(
                        height=500,
                        bargap=0.1,
                        xaxis_title=col,
                        yaxis_title="Count"
                    )
                    
                    plots.append({
                        "id": f"plot_hist_{idx}",
                        "figure": fig.to_json(),
                        "title": f"Distribution of {col}",
                        "insight": viz.get("description"),
                        "plot_type": "histogram"
                    })
                    break
                except Exception as e:
                    if self.debug:
                        print(f"Error creating histogram for {col}: {str(e)}")
                        
    def _add_bar_plot(self, df, cat_cols, desc, idx, plots, viz):
        """Helper method to add a bar plot."""
        for col in cat_cols:
            if col.lower() in desc.lower():
                try:
                    # Get column with second highest mention in description for possible grouping
                    group_col = None
                    if len(cat_cols) > 1:
                        for potential_group in cat_cols:
                            if potential_group != col and potential_group.lower() in desc.lower():
                                group_col = potential_group
                                break
                    
                    # Use grouping if found and makes sense
                    if group_col and df[group_col].nunique() <= 5 and df[col].nunique() <= 10:
                        # Create grouped bar chart
                        grouped_data = pd.crosstab(df[col], df[group_col])
                        fig = px.bar(
                            grouped_data, 
                            barmode='group',
                            title=f"{col} by {group_col}",
                            template='plotly_white',
                            color_discrete_sequence=px.colors.qualitative.Plotly
                        )
                    else:
                        # Create regular bar chart of top values
                        top_values = df[col].value_counts().head(10)
                        fig = px.bar(
                            x=top_values.index, 
                            y=top_values.values,
                            title=f"Top values for {col}",
                            template='plotly_white',
                            color_discrete_sequence=['#636EFA']
                        )
                        fig.update_xaxes(title=col)
                        fig.update_yaxes(title="Count")
                    
                    fig.update_layout(height=500)
                    
                    plots.append({
                        "id": f"plot_bar_{idx}",
                        "figure": fig.to_json(),
                        "title": f"Bar Chart of {col}" + (f" by {group_col}" if group_col else ""),
                        "insight": viz.get("description"),
                        "plot_type": "bar"
                    })
                    break
                except Exception as e:
                    if self.debug:
                        print(f"Error creating bar chart for {col}: {str(e)}")
                        
    def _add_pie_plot(self, df, cat_cols, desc, idx, plots, viz):
        """Helper method to add a pie plot."""
        for col in cat_cols:
            if col.lower() in desc.lower() and df[col].nunique() <= 10:
                try:
                    values = df[col].value_counts()
                    
                    # If too many categories, group smaller ones as "Other"
                    if len(values) > 8:
                        top_n = values.head(7)
                        other_sum = values.iloc[7:].sum()
                        values = pd.concat([top_n, pd.Series({"Other": other_sum})])
                    
                    fig = px.pie(
                        values=values.values,
                        names=values.index,
                        title=f"Composition of {col}",
                        template='plotly_white',
                        color_discrete_sequence=px.colors.qualitative.Plotly,
                        hole=0.3  # Create a donut chart for better aesthetics
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(height=500)
                    
                    plots.append({
                        "id": f"plot_pie_{idx}",
                        "figure": fig.to_json(),
                        "title": f"Composition of {col}",
                        "insight": viz.get("description"),
                        "plot_type": "pie"
                    })
                    break
                except Exception as e:
                    if self.debug:
                        print(f"Error creating pie chart for {col}: {str(e)}")
                        
    def _add_scatter_plot(self, df, num_cols, cat_cols, desc, idx, plots, viz):
        """Helper method to add a scatter plot."""
        x_col = None
        y_col = None
        
        # Try to find two numeric columns mentioned
        for col in num_cols:
            if col.lower() in desc.lower():
                if not x_col:
                    x_col = col
                elif not y_col:
                    y_col = col
                    break
        
        if x_col and y_col:
            try:
                # Look for a potential color column
                color_col = None
                for col in cat_cols:
                    if col.lower() in desc.lower() and df[col].nunique() <= 10:
                        color_col = col
                        break
                
                # Create scatter plot with or without color grouping
                if color_col:
                    fig = px.scatter(
                        df, 
                        x=x_col, 
                        y=y_col, 
                        color=color_col,
                        title=f"{y_col} vs {x_col} by {color_col}",
                        template='plotly_white',
                        opacity=0.7,
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                else:
                    fig = px.scatter(
                        df, 
                        x=x_col, 
                        y=y_col,
                        title=f"{y_col} vs {x_col}",
                        template='plotly_white',
                        opacity=0.7,
                        color_discrete_sequence=['#636EFA']
                    )
                
                # Add trendline if no color grouping
                if not color_col:
                    fig.update_layout(
                        shapes=[{
                            'type': 'line',
                            'x0': df[x_col].min(),
                            'y0': df[y_col].min(),
                            'x1': df[x_col].max(),
                            'y1': df[y_col].max(),
                            'line': {
                                'color': 'rgba(255, 0, 0, 0.3)',
                                'width': 2,
                                'dash': 'dot',
                            }
                        }]
                    )
                    
                fig.update_layout(height=500)
                
                plots.append({
                    "id": f"plot_scatter_{idx}",
                    "figure": fig.to_json(),
                    "title": f"Relationship between {y_col} and {x_col}" + (f" by {color_col}" if color_col else ""),
                    "insight": viz.get("description"),
                    "plot_type": "scatter"
                })
            except Exception as e:
                if self.debug:
                    print(f"Error creating scatter plot for {x_col} vs {y_col}: {str(e)}")
                    
    def _add_box_plot(self, df, num_cols, cat_cols, desc, idx, plots, viz):
        """Helper method to add a box plot."""
        y_col = None
        x_col = None
        
        # Find a numeric column first
        for col in num_cols:
            if col.lower() in desc.lower():
                y_col = col
                break
        
        # Then try to find a categorical column
        for col in cat_cols:
            if col.lower() in desc.lower() and df[col].nunique() <= 10:
                x_col = col
                break
        
        if y_col:
            try:
                if x_col:
                    fig = px.box(
                        df,
                        y=y_col,
                        x=x_col,
                        title=f"Distribution of {y_col} by {x_col}",
                        template='plotly_white',
                        color=x_col,
                        notched=True,  # Add notches for better comparison
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                else:
                    fig = px.box(
                        df,
                        y=y_col,
                        title=f"Box Plot of {y_col}",
                        template='plotly_white',
                        color_discrete_sequence=['#636EFA']
                    )
                
                fig.update_layout(height=500)
                
                plots.append({
                    "id": f"plot_box_{idx}",
                    "figure": fig.to_json(),
                    "title": f"Box Plot of {y_col}" + (f" by {x_col}" if x_col else ""),
                    "insight": viz.get("description"),
                    "plot_type": "box"
                })
            except Exception as e:
                if self.debug:
                    print(f"Error creating box plot for {y_col}: {str(e)}")
    
    def _add_line_plot(self, df, num_cols, date_cols, cat_cols, desc, idx, plots, viz):
        """Helper method to add a line plot."""
        y_col = None
        x_col = None
        
        # First find a numeric column for y-axis
        for col in num_cols:
            if col.lower() in desc.lower():
                y_col = col
                break
        
        # Then look for a date column for x-axis
        for col in date_cols:
            if col.lower() in desc.lower():
                x_col = col
                break
                
        # If no date column, look for a categorical column for x-axis
        if not x_col:
            for col in cat_cols:
                if col.lower() in desc.lower():
                    x_col = col
                    break
        
        # If still no x_col, use the first date column if available
        if not x_col and date_cols:
            x_col = date_cols[0]
            
        # If still no x_col, use index
        if not x_col:
            # Just create a sequence index
            x_col = "index"
        
        if y_col:
            try:
                # Look for a grouping column
                group_col = None
                for col in cat_cols:
                    if col != x_col and col.lower() in desc.lower() and df[col].nunique() <= 5:
                        group_col = col
                        break
                
                if x_col == "index":
                    # Create a copy of the dataframe with an explicit index column
                    plot_df = df.copy()
                    plot_df["index"] = range(len(df))
                else:
                    plot_df = df.copy()
                    
                    # If x_col is a date/time, ensure it's properly formatted
                    if x_col in date_cols:
                        try:
                            plot_df[x_col] = pd.to_datetime(plot_df[x_col])
                            # Sort by date
                            plot_df = plot_df.sort_values(by=x_col)
                        except:
                            pass
                
                # Create line chart with or without grouping
                if group_col:
                    fig = px.line(
                        plot_df,
                        x=x_col,
                        y=y_col,
                        color=group_col,
                        title=f"Trend of {y_col} by {x_col} grouped by {group_col}",
                        template='plotly_white',
                        markers=True,  # Add markers to the lines
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                else:
                    fig = px.line(
                        plot_df,
                        x=x_col,
                        y=y_col,
                        title=f"Trend of {y_col} by {x_col}",
                        template='plotly_white',
                        markers=True,  # Add markers to the lines
                        color_discrete_sequence=['#636EFA']
                    )
                
                fig.update_layout(height=500)
                fig.update_traces(line=dict(width=3))  # Make lines thicker
                
                plots.append({
                    "id": f"plot_line_{idx}",
                    "figure": fig.to_json(),
                    "title": f"Trend of {y_col} by {x_col}" + (f" grouped by {group_col}" if group_col else ""),
                    "insight": viz.get("description"),
                    "plot_type": "line"
                })
            except Exception as e:
                if self.debug:
                    print(f"Error creating line plot for {y_col}: {str(e)}")
    
    def _add_area_plot(self, df, num_cols, date_cols, cat_cols, desc, idx, plots, viz):
        """Helper method to add an area plot."""
        y_col = None
        x_col = None
        
        # First find a numeric column for y-axis
        for col in num_cols:
            if col.lower() in desc.lower():
                y_col = col
                break
        
        # Then look for a date column for x-axis
        for col in date_cols:
            if col.lower() in desc.lower():
                x_col = col
                break
                
        # If no date column, look for a categorical column for x-axis
        if not x_col:
            for col in cat_cols:
                if col.lower() in desc.lower():
                    x_col = col
                    break
        
        # If still no x_col, use the first date column if available
        if not x_col and date_cols:
            x_col = date_cols[0]
            
        # If still no x_col, use index
        if not x_col:
            # Just create a sequence index
            x_col = "index"
        
        if y_col:
            try:
                # Look for a grouping column
                group_col = None
                for col in cat_cols:
                    if col != x_col and col.lower() in desc.lower() and df[col].nunique() <= 5:
                        group_col = col
                        break
                
                if x_col == "index":
                    # Create a copy of the dataframe with an explicit index column
                    plot_df = df.copy()
                    plot_df["index"] = range(len(df))
                else:
                    plot_df = df.copy()
                    
                    # If x_col is a date/time, ensure it's properly formatted
                    if x_col in date_cols:
                        try:
                            plot_df[x_col] = pd.to_datetime(plot_df[x_col])
                            # Sort by date
                            plot_df = plot_df.sort_values(by=x_col)
                        except:
                            pass
                
                # Create area chart with or without grouping
                if group_col and "stack" in desc.lower():
                    # Stacked area chart
                    fig = px.area(
                        plot_df,
                        x=x_col,
                        y=y_col,
                        color=group_col,
                        title=f"Stacked Area Chart of {y_col} by {x_col} grouped by {group_col}",
                        template='plotly_white',
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                elif group_col:
                    # Grouped area chart
                    fig = px.area(
                        plot_df,
                        x=x_col,
                        y=y_col,
                        color=group_col,
                        title=f"Area Chart of {y_col} by {x_col} grouped by {group_col}",
                        template='plotly_white',
                        color_discrete_sequence=px.colors.qualitative.Plotly,
                        groupnorm='fraction'  # Normalize to show percentage
                    )
                else:
                    # Simple area chart
                    fig = px.area(
                        plot_df,
                        x=x_col,
                        y=y_col,
                        title=f"Area Chart of {y_col} by {x_col}",
                        template='plotly_white',
                        color_discrete_sequence=['#636EFA']
                    )
                
                fig.update_layout(height=500)
                
                plots.append({
                    "id": f"plot_area_{idx}",
                    "figure": fig.to_json(),
                    "title": f"Area Chart of {y_col} by {x_col}" + (f" grouped by {group_col}" if group_col else ""),
                    "insight": viz.get("description"),
                    "plot_type": "area"
                })
            except Exception as e:
                if self.debug:
                    print(f"Error creating area plot for {y_col}: {str(e)}")
    
    def _add_heatmap_plot(self, df, num_cols, desc, idx, plots, viz):
        """Helper method to add a heatmap."""
        try:
            # Create correlation heatmap for numeric columns
            if len(num_cols) >= 2:
                # Calculate correlation matrix
                corr_matrix = df[num_cols].corr()
                
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu_r',  # Red-Blue diverging colorscale
                    zmid=0,  # Center the colorscale at 0
                    text=corr_matrix.round(2).values,
                    texttemplate="%{text}",
                    hovertemplate="Correlation between %{x} and %{y}: %{z:.2f}<extra></extra>"
                ))
                
                fig.update_layout(
                    title="Correlation Heatmap of Numeric Features",
                    template='plotly_white',
                    height=600,
                    width=700
                )
                
                plots.append({
                    "id": f"plot_heatmap_{idx}",
                    "figure": fig.to_json(),
                    "title": "Correlation Heatmap",
                    "insight": viz.get("description"),
                    "plot_type": "heatmap"
                })
        except Exception as e:
            if self.debug:
                print(f"Error creating heatmap: {str(e)}")
    
    def _add_bubble_plot(self, df, num_cols, cat_cols, desc, idx, plots, viz):
        """Helper method to add a bubble plot."""
        if len(num_cols) >= 3:
            try:
                # We need at least 3 numeric columns
                x_col = num_cols[0]
                y_col = num_cols[1]
                size_col = num_cols[2]
                
                # Find mentioned columns
                for col in num_cols:
                    if col.lower() in desc.lower():
                        if not 'x_col' in locals() or not x_col:
                            x_col = col
                        elif not 'y_col' in locals() or not y_col:
                            y_col = col
                        elif not 'size_col' in locals() or not size_col:
                            size_col = col
                            break
                
                # Look for a color column
                color_col = None
                for col in cat_cols:
                    if col.lower() in desc.lower() and df[col].nunique() <= 10:
                        color_col = col
                        break
                
                # Set size reference to make bubbles appropriately sized
                size_ref = df[size_col].max() / 1000
                
                # Create bubble chart
                if color_col:
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        size=size_col,
                        color=color_col,
                        title=f"Bubble Chart: {y_col} vs {x_col} (Size: {size_col}, Color: {color_col})",
                        template='plotly_white',
                        size_max=30,  # Maximum bubble size
                        opacity=0.7,
                        color_discrete_sequence=px.colors.qualitative.Plotly,
                        hover_name=color_col if df[color_col].nunique() <= 20 else None,
                        hover_data=[size_col]
                    )
                else:
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        size=size_col,
                        title=f"Bubble Chart: {y_col} vs {x_col} (Size: {size_col})",
                        template='plotly_white',
                        size_max=30,  # Maximum bubble size
                        opacity=0.7,
                        color_discrete_sequence=['#636EFA'],
                        hover_data=[size_col]
                    )
                
                fig.update_layout(height=500)
                
                plots.append({
                    "id": f"plot_bubble_{idx}",
                    "figure": fig.to_json(),
                    "title": f"Bubble Chart: {y_col} vs {x_col}",
                    "insight": viz.get("description"),
                    "plot_type": "bubble"
                })
            except Exception as e:
                if self.debug:
                    print(f"Error creating bubble plot: {str(e)}")
    
    def _add_violin_plot(self, df, num_cols, cat_cols, desc, idx, plots, viz):
        """Helper method to add a violin plot."""
        y_col = None
        x_col = None
        
        # Find a numeric column first for y-axis
        for col in num_cols:
            if col.lower() in desc.lower():
                y_col = col
                break
        
        # Then try to find a categorical column for x-axis
        for col in cat_cols:
            if col.lower() in desc.lower() and df[col].nunique() <= 10:
                x_col = col
                break
        
        if y_col:
            try:
                if x_col:
                    # Create violin plot with categories
                    fig = px.violin(
                        df,
                        y=y_col,
                        x=x_col,
                        color=x_col,
                        box=True,  # Add box plot inside violin
                        points="all",  # Show all points
                        title=f"Distribution of {y_col} by {x_col}",
                        template='plotly_white',
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                else:
                    # Create simple violin plot
                    fig = px.violin(
                        df,
                        y=y_col,
                        box=True,  # Add box plot inside violin
                        points="all",  # Show all points
                        title=f"Violin Plot of {y_col}",
                        template='plotly_white',
                        color_discrete_sequence=['#636EFA']
                    )
                
                fig.update_layout(height=500)
                
                plots.append({
                    "id": f"plot_violin_{idx}",
                    "figure": fig.to_json(),
                    "title": f"Violin Plot of {y_col}" + (f" by {x_col}" if x_col else ""),
                    "insight": viz.get("description"),
                    "plot_type": "violin"
                })
            except Exception as e:
                if self.debug:
                    print(f"Error creating violin plot for {y_col}: {str(e)}")
    
    def _add_radar_plot(self, df, num_cols, cat_cols, desc, idx, plots, viz):
        """Helper method to add a radar chart."""
        # Need at least one categorical and multiple numeric columns
        if len(num_cols) >= 3 and len(cat_cols) >= 1:
            try:
                # Find the categorical column for grouping
                group_col = None
                for col in cat_cols:
                    if col.lower() in desc.lower() and df[col].nunique() <= 10:
                        group_col = col
                        break
                
                if not group_col:
                    group_col = cat_cols[0]
                
                # Limit to reasonable number of categories
                if df[group_col].nunique() > 6:
                    top_categories = df[group_col].value_counts().head(6).index.tolist()
                    radar_df = df[df[group_col].isin(top_categories)].copy()
                else:
                    radar_df = df.copy()
                
                # Use up to 5 numeric columns
                features = []
                for col in num_cols:
                    if col.lower() in desc.lower():
                        features.append(col)
                
                if len(features) < 3:
                    features = num_cols[:5]  # Take first 5 numeric columns
                else:
                    features = features[:5]  # Limit to 5 features
                
                # Calculate mean of each feature for each category
                radar_data = radar_df.groupby(group_col)[features].mean().reset_index()
                
                # Create radar chart
                fig = go.Figure()
                
                # Normalize features to 0-1 scale for better radar chart
                for feature in features:
                    min_val = radar_df[feature].min()
                    max_val = radar_df[feature].max()
                    if max_val > min_val:  # Avoid division by zero
                        radar_data[f"{feature}_norm"] = (radar_data[feature] - min_val) / (max_val - min_val)
                    else:
                        radar_data[f"{feature}_norm"] = 0
                
                # Add each category as a trace
                for i, category in enumerate(radar_data[group_col]):
                    values = radar_data.loc[radar_data[group_col] == category, [f"{f}_norm" for f in features]].values.flatten().tolist()
                    # Close the loop by repeating the first value
                    values.append(values[0])
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=features + [features[0]],  # Close the loop
                        fill='toself',
                        name=str(category),
                        line_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    title=f"Radar Chart of {', '.join(features)} by {group_col}",
                    template='plotly_white',
                    height=600
                )
                
                plots.append({
                    "id": f"plot_radar_{idx}",
                    "figure": fig.to_json(),
                    "title": f"Radar Chart by {group_col}",
                    "insight": viz.get("description"),
                    "plot_type": "radar"
                })
            except Exception as e:
                if self.debug:
                    print(f"Error creating radar plot: {str(e)}")
    
    def _add_treemap_plot(self, df, cat_cols, num_cols, desc, idx, plots, viz):
        """Helper method to add a treemap."""
        if len(cat_cols) >= 1:
            try:
                # Find categorical columns for hierarchy
                hierarchy_cols = []
                for col in cat_cols:
                    if col.lower() in desc.lower() and df[col].nunique() <= 20:
                        hierarchy_cols.append(col)
                
                # If no specific columns found, use the first 1-2 categorical columns
                if not hierarchy_cols:
                    hierarchy_cols = [cat_cols[0]]
                    if len(cat_cols) >= 2:
                        hierarchy_cols.append(cat_cols[1])
                
                # Find value column (numeric)
                value_col = None
                for col in num_cols:
                    if col.lower() in desc.lower():
                        value_col = col
                        break
                
                # If no specific value column found, use the first numeric column
                if not value_col and num_cols:
                    value_col = num_cols[0]
                
                # Create treemap
                if value_col:
                    fig = px.treemap(
                        df,
                        path=hierarchy_cols,
                        values=value_col,
                        title=f"Treemap of {' > '.join(hierarchy_cols)} by {value_col}",
                        template='plotly_white',
                        color=value_col,
                        color_continuous_scale='RdBu',
                        color_continuous_midpoint=df[value_col].median()
                    )
                else:
                    # Count-based treemap if no value column
                    fig = px.treemap(
                        df,
                        path=hierarchy_cols,
                        title=f"Treemap of {' > '.join(hierarchy_cols)}",
                        template='plotly_white'
                    )
                
                fig.update_layout(height=600)
                fig.update_traces(
                    textinfo="label+value+percent parent+percent root",
                    hovertemplate='<b>%{label}</b><br>Value: %{value}<br>Percentage: %{percentRoot:.1%}<extra></extra>'
                )
                
                plots.append({
                    "id": f"plot_treemap_{idx}",
                    "figure": fig.to_json(),
                    "title": f"Treemap of {' > '.join(hierarchy_cols)}" + (f" by {value_col}" if value_col else ""),
                    "insight": viz.get("description"),
                    "plot_type": "treemap"
                })
            except Exception as e:
                if self.debug:
                    print(f"Error creating treemap: {str(e)}")
    
    def _create_standard_visualizations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create an enhanced set of standard visualizations as a reliable fallback.
        """
        plots = []
        viz_data = {"success": False, "plots": [], "debug_info": {}, "is_fallback": True}
        
        try:
            # Get column types
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            
            # Try to detect string columns that might be dates
            potential_date_cols = []
            for col in cat_cols:
                try:
                    # Try to convert to datetime with common formats
                    test = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
                    if test.notna().mean() < 0.8:
                        test = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')
                    if test.notna().mean() < 0.8:
                        test = pd.to_datetime(df[col], format='%m/%d/%Y', errors='coerce')
                    if test.notna().mean() < 0.8:
                        test = pd.to_datetime(df[col], format='%Y/%m/%d', errors='coerce')
                    
                    # If at least 80% of values can be converted, consider it a date column
                    if test.notna().mean() >= 0.8:
                        potential_date_cols.append(col)
                        # Also convert it for use in visualizations
                        df[col + '_date'] = test
                        date_cols.append(col + '_date')
                except:
                    pass
            
            # 1. Histogram for first numeric column
            if num_cols:
                try:
                    col = num_cols[0]
                    fig = px.histogram(
                        df, 
                        x=col, 
                        title=f"Distribution of {col}",
                        template='plotly_white',
                        opacity=0.8,
                        color_discrete_sequence=['#636EFA']
                    )
                    fig.update_layout(
                        height=500,
                        bargap=0.1
                    )
                    
                    plots.append({
                        "id": "plot_hist",
                        "figure": fig.to_json(),
                        "title": f"Distribution of {col}",
                        "insight": f"This histogram shows the distribution of values for {col}.",
                        "plot_type": "histogram"
                    })
                except Exception as e:
                    if self.debug:
                        print(f"Error creating histogram: {str(e)}")
            
            # 2. Bar chart for first categorical column
            if cat_cols:
                try:
                    col = cat_cols[0]
                    # Handle case where values might be empty or all null
                    if df[col].count() > 0:
                        top_values = df[col].value_counts().head(10)
                        fig = px.bar(
                            x=top_values.index, 
                            y=top_values.values,
                            title=f"Top values for {col}",
                            template='plotly_white',
                            color_discrete_sequence=['#636EFA']
                        )
                        fig.update_layout(height=500)
                        
                        plots.append({
                            "id": "plot_bar",
                            "figure": fig.to_json(),
                            "title": f"Top values for {col}",
                            "insight": f"This bar chart shows the most common values for {col}.",
                            "plot_type": "bar"
                        })
                except Exception as e:
                    if self.debug:
                        print(f"Error creating bar chart: {str(e)}")
            
            # 3. Pie chart if categorical column has reasonable cardinality
            if cat_cols:
                try:
                    for col in cat_cols:
                        if 2 <= df[col].nunique() <= 10 and df[col].count() > 0:
                            values = df[col].value_counts()
                            fig = px.pie(
                                values=values.values,
                                names=values.index,
                                title=f"Composition of {col}",
                                template='plotly_white',
                                hole=0.3,  # Create a donut chart
                                color_discrete_sequence=px.colors.qualitative.Plotly
                            )
                            fig.update_traces(textposition='inside', textinfo='percent+label')
                            fig.update_layout(height=500)
                            
                            plots.append({
                                "id": "plot_pie",
                                "figure": fig.to_json(),
                                "title": f"Composition of {col}",
                                "insight": f"This pie chart shows the proportion of each {col} category.",
                                "plot_type": "pie"
                            })
                            break  # Only create one pie chart
                except Exception as e:
                    if self.debug:
                        print(f"Error creating pie chart: {str(e)}")
            
            # 4. Scatter plot if we have at least 2 numeric columns
            if len(num_cols) >= 2:
                try:
                    x_col, y_col = num_cols[0], num_cols[1]
                    
                    # Try to find a good categorical column for color
                    color_col = None
                    for col in cat_cols:
                        if 2 <= df[col].nunique() <= 6:
                            color_col = col
                            break
                    
                    if color_col:
                        fig = px.scatter(
                            df, 
                            x=x_col, 
                            y=y_col,
                            color=color_col,
                            title=f"{y_col} vs {x_col} by {color_col}",
                            template='plotly_white',
                            opacity=0.7,
                            color_discrete_sequence=px.colors.qualitative.Plotly
                        )
                    else:
                        fig = px.scatter(
                            df, 
                            x=x_col, 
                            y=y_col,
                            title=f"{y_col} vs {x_col}",
                            template='plotly_white',
                            opacity=0.7,
                            color_discrete_sequence=['#636EFA']
                        )
                        
                    fig.update_layout(height=500)
                    
                    plots.append({
                        "id": "plot_scatter",
                        "figure": fig.to_json(),
                        "title": f"Relationship between {y_col} and {x_col}" + (f" by {color_col}" if color_col else ""),
                        "insight": f"This scatter plot shows the relationship between {y_col} and {x_col}.",
                        "plot_type": "scatter"
                    })
                except Exception as e:
                    if self.debug:
                        print(f"Error creating scatter plot: {str(e)}")
            
            # 5. Box plot for first numeric column with categorical grouping
            if num_cols and cat_cols:
                try:
                    y_col = num_cols[0]
                    
                    # Find a categorical column with reasonable cardinality
                    x_col = None
                    for col in cat_cols:
                        if 2 <= df[col].nunique() <= 6:
                            x_col = col
                            break
                    
                    if x_col:
                        fig = px.box(
                            df,
                            y=y_col,
                            x=x_col,
                            color=x_col,
                            title=f"Distribution of {y_col} by {x_col}",
                            template='plotly_white',
                            notched=True,
                            color_discrete_sequence=px.colors.qualitative.Plotly
                        )
                    else:
                        fig = px.box(
                            df,
                            y=y_col,
                            title=f"Box Plot of {y_col}",
                            template='plotly_white',
                            color_discrete_sequence=['#636EFA']
                        )
                    
                    fig.update_layout(height=500)
                    
                    plots.append({
                        "id": "plot_box",
                        "figure": fig.to_json(),
                        "title": f"Box Plot of {y_col}" + (f" by {x_col}" if x_col else ""),
                        "insight": f"This box plot shows the distribution and outliers of {y_col}.",
                        "plot_type": "box"
                    })
                except Exception as e:
                    if self.debug:
                        print(f"Error creating box plot: {str(e)}")
            
            # 6. Line chart if we have date columns or potential date columns
            combined_date_cols = date_cols + potential_date_cols
            if num_cols and combined_date_cols:
                try:
                    y_col = num_cols[0]
                    x_col = combined_date_cols[0]
                    
                    # Create a copy of the dataframe to work with
                    line_df = df.copy()
                    
                    # Convert the column to datetime if it's a potential date column
                    if x_col in potential_date_cols:
                        try:
                            line_df[x_col] = pd.to_datetime(line_df[x_col], errors='coerce')
                            # Drop NaT values
                            line_df = line_df.dropna(subset=[x_col])
                            # Sort by date
                            line_df = line_df.sort_values(by=x_col)
                        except:
                            # If conversion fails, skip this visualization
                            raise ValueError(f"Failed to convert {x_col} to datetime")
                    
                    # Find a potential grouping column
                    group_col = None
                    for col in cat_cols:
                        if 2 <= df[col].nunique() <= 5:
                            group_col = col
                            break
                    
                    if group_col:
                        fig = px.line(
                            line_df,
                            x=x_col,
                            y=y_col,
                            color=group_col,
                            title=f"Trend of {y_col} over {x_col} grouped by {group_col}",
                            template='plotly_white',
                            markers=True,
                            color_discrete_sequence=px.colors.qualitative.Plotly
                        )
                    else:
                        fig = px.line(
                            line_df,
                            x=x_col,
                            y=y_col,
                            title=f"Trend of {y_col} over {x_col}",
                            template='plotly_white',
                            markers=True,
                            color_discrete_sequence=['#636EFA']
                        )
                    
                    fig.update_layout(height=500)
                    fig.update_traces(line=dict(width=3))
                    
                    plots.append({
                        "id": "plot_line",
                        "figure": fig.to_json(),
                        "title": f"Trend of {y_col} over time" + (f" by {group_col}" if group_col else ""),
                        "insight": f"This line chart shows how {y_col} changes over time.",
                        "plot_type": "line"
                    })
                except Exception as e:
                    if self.debug:
                        print(f"Error creating line chart: {str(e)}")
            
            # 7. Correlation heatmap for numeric columns
            if len(num_cols) >= 3:
                try:
                    # Calculate correlation matrix
                    corr_matrix = df[num_cols].corr()
                    
                    # Create heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu_r',
                        zmid=0,
                        text=corr_matrix.round(2).values,
                        texttemplate="%{text}",
                        hovertemplate="Correlation between %{x} and %{y}: %{z:.2f}<extra></extra>"
                    ))
                    
                    fig.update_layout(
                        title="Correlation Heatmap of Numeric Features",
                        template='plotly_white',
                        height=600,
                        width=700
                    )
                    
                    plots.append({
                        "id": "plot_heatmap",
                        "figure": fig.to_json(),
                        "title": "Correlation Heatmap",
                        "insight": "This heatmap shows the correlation between numeric variables. Strong positive correlations appear in red, and strong negative correlations appear in blue.",
                        "plot_type": "heatmap"
                    })
                except Exception as e:
                    if self.debug:
                        print(f"Error creating heatmap: {str(e)}")
            
            # Add plots to result
            viz_data["plots"] = plots
            viz_data["success"] = len(plots) > 0
            
            return viz_data
            
        except Exception as e:
            if self.debug:
                print(f"Error in _create_standard_visualizations: {str(e)}")
                print(traceback.format_exc())
                viz_data["debug_info"]["fallback_error"] = str(e)
                viz_data["debug_info"]["fallback_traceback"] = traceback.format_exc()
            
            # Return empty result if even the fallback fails
            viz_data["plots"] = []
            viz_data["success"] = False
            
            return viz_data