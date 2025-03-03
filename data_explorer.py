import logging
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

from agent_framework_core import (
    BaseAgent, 
    AgentMemory, 
    Message, 
    AgentState,
    DatasetInfo
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class ColumnProfile:
    """Profile information for a single column in a dataset."""
    name: str
    dtype: str
    count: int
    missing_count: int
    missing_percentage: float
    unique_count: Optional[int] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std_dev: Optional[float] = None
    quartile_1: Optional[float] = None
    quartile_3: Optional[float] = None
    iqr: Optional[float] = None
    outliers_count: Optional[int] = None
    top_values: Optional[List[Tuple[Any, int]]] = None
    histogram_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary representation."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "count": self.count,
            "missing_count": self.missing_count,
            "missing_percentage": self.missing_percentage,
            "unique_count": self.unique_count,
            "min_value": self.min_value if not pd.isna(self.min_value) else None,
            "max_value": self.max_value if not pd.isna(self.max_value) else None,
            "mean": self.mean if not pd.isna(self.mean) else None,
            "median": self.median if not pd.isna(self.median) else None,
            "std_dev": self.std_dev if not pd.isna(self.std_dev) else None,
            "quartile_1": self.quartile_1 if not pd.isna(self.quartile_1) else None,
            "quartile_3": self.quartile_3 if not pd.isna(self.quartile_3) else None,
            "iqr": self.iqr if not pd.isna(self.iqr) else None,
            "outliers_count": self.outliers_count,
            "top_values": self.top_values,
            "histogram_data": self.histogram_data
        }

@dataclass
class DataProfile:
    """Complete profile for a dataset including all columns and dataset-level metrics."""
    dataset_name: str
    row_count: int
    column_count: int
    memory_usage: float  # in MB
    duplicate_rows_count: int
    column_profiles: Dict[str, ColumnProfile] = field(default_factory=dict)
    correlations: Optional[Dict[str, Dict[str, float]]] = None
    timestamp: float = field(default_factory=lambda: pd.Timestamp.now().timestamp())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary representation."""
        return {
            "dataset_name": self.dataset_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "memory_usage": self.memory_usage,
            "duplicate_rows_count": self.duplicate_rows_count,
            "column_profiles": {name: profile.to_dict() for name, profile in self.column_profiles.items()},
            "correlations": self.correlations,
            "timestamp": self.timestamp
        }
    
    def get_column_names(self) -> List[str]:
        """Get list of all column names."""
        return list(self.column_profiles.keys())
    
    def get_numeric_columns(self) -> List[str]:
        """Get list of numeric column names."""
        return [name for name, profile in self.column_profiles.items() 
                if profile.dtype in ['int64', 'float64', 'int32', 'float32']]
    
    def get_categorical_columns(self) -> List[str]:
        """Get list of categorical column names."""
        return [name for name, profile in self.column_profiles.items() 
                if profile.dtype in ['object', 'category', 'bool']]
    
    def get_datetime_columns(self) -> List[str]:
        """Get list of datetime column names."""
        return [name for name, profile in self.column_profiles.items() 
                if profile.dtype in ['datetime64[ns]', 'timedelta64[ns]']]
    
    def get_high_missing_columns(self, threshold: float = 0.2) -> List[str]:
        """Get columns with missing percentage above threshold."""
        return [name for name, profile in self.column_profiles.items() 
                if profile.missing_percentage > threshold]
    
    def get_high_cardinality_columns(self, threshold: float = 0.8) -> List[str]:
        """Get columns with high cardinality (unique values ratio above threshold)."""
        return [name for name, profile in self.column_profiles.items() 
                if profile.unique_count is not None and 
                (profile.unique_count / profile.count > threshold)]
    
    def get_columns_with_outliers(self) -> List[str]:
        """Get columns with outliers."""
        return [name for name, profile in self.column_profiles.items() 
                if profile.outliers_count is not None and profile.outliers_count > 0]

class DataExplorer(BaseAgent):
    """
    Agent that analyzes raw data to generate profiles and basic statistical summaries.
    
    This agent performs data quality checks, detects missing values, outliers, 
    distributions, and summarizes overall structure.
    """
    
    def __init__(self, agent_id: str, memory: AgentMemory):
        """
        Initialize the Data Explorer agent.
        
        Args:
            agent_id: Unique identifier for this agent
            memory: Shared memory system for agent communication
        """
        super().__init__(agent_id, memory)
        self.logger = logging.getLogger(f"DataExplorer_{agent_id}")
        
    def _execute(self, action: str = "explore_data", **kwargs) -> Any:
        """
        Execute the specified action.
        
        Args:
            action: The action to execute
            kwargs: Additional arguments specific to the action
            
        Returns:
            The result of the action
        """
        if action == "explore_data":
            return self.explore_data(**kwargs)
        elif action == "generate_correlation_matrix":
            return self.generate_correlation_matrix(**kwargs)
        elif action == "detect_outliers":
            return self.detect_outliers(**kwargs)
        elif action == "get_column_profile":
            return self.get_column_profile(**kwargs)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def explore_data(self, 
                    dataframe: Optional[pd.DataFrame] = None,
                    dataset_info: Optional[DatasetInfo] = None,
                    spreadsheet_id: Optional[str] = None,
                    sheet_name: Optional[str] = None,
                    include_correlations: bool = True,
                    include_histograms: bool = True,
                    top_n_values: int = 5,
                    outlier_detection_method: str = "iqr",
                    outlier_threshold: float = 1.5) -> Dict[str, Any]:
        """
        Explore and profile a dataset.
        
        Args:
            dataframe: Pandas DataFrame to profile
            dataset_info: DatasetInfo object
            spreadsheet_id: ID of the Google Sheet (if data retrieved from there)
            sheet_name: Name of the sheet (if data retrieved from there)
            include_correlations: Whether to include correlation matrix
            include_histograms: Whether to include histogram data for numerical columns
            top_n_values: Number of top values to include for categorical columns
            outlier_detection_method: Method to use for outlier detection ('iqr' or 'zscore')
            outlier_threshold: Threshold for outlier detection
            
        Returns:
            Dictionary with exploration results
        """
        self.state = AgentState.RUNNING
        self.logger.info("Starting data exploration")
        
        if dataframe is None:
            self.logger.error("No dataframe provided for exploration")
            self.state = AgentState.FAILED
            return {"error": "No dataframe provided for exploration"}
        
        # Set default dataset name based on available information
        dataset_name = "unknown_dataset"
        if dataset_info and dataset_info.name:
            dataset_name = dataset_info.name
        elif spreadsheet_id and sheet_name:
            dataset_name = f"{spreadsheet_id}_{sheet_name}"
        
        try:
            # Create the dataset profile
            profile = self._create_data_profile(
                dataframe, 
                dataset_name, 
                include_correlations, 
                include_histograms, 
                top_n_values,
                outlier_detection_method, 
                outlier_threshold
            )
            
            # Store the profile in memory
            memory_key = f"data_profile:{dataset_name}"
            self.memory.store(memory_key, profile.to_dict())
            
            # Generate a summary report
            summary_report = self._generate_summary_report(profile)
            memory_key_summary = f"data_summary:{dataset_name}"
            self.memory.store(memory_key_summary, summary_report)
            
            # Notify that exploration is complete
            self.send_message("master_planner", "exploration_complete", {
                "dataset_name": dataset_name,
                "profile_key": memory_key,
                "summary_key": memory_key_summary
            })
            
            self.state = AgentState.COMPLETED
            self.logger.info(f"Data exploration completed for dataset: {dataset_name}")
            
            # Return exploration results
            return {
                "profile": profile.to_dict(),
                "summary": summary_report,
                "dataset_name": dataset_name
            }
            
        except Exception as e:
            self.logger.error(f"Error during data exploration: {str(e)}")
            self.state = AgentState.FAILED
            return {"error": f"Data exploration failed: {str(e)}"}
    
    def _create_data_profile(self, 
                            df: pd.DataFrame, 
                            dataset_name: str,
                            include_correlations: bool = True,
                            include_histograms: bool = True,
                            top_n_values: int = 5,
                            outlier_detection_method: str = "iqr",
                            outlier_threshold: float = 1.5) -> DataProfile:
        """
        Create a comprehensive profile of the dataset.
        
        Args:
            df: Pandas DataFrame to profile
            dataset_name: Name of the dataset
            include_correlations: Whether to include correlation matrix
            include_histograms: Whether to include histogram data
            top_n_values: Number of top values to include for categorical columns
            outlier_detection_method: Method for outlier detection
            outlier_threshold: Threshold for outlier detection
            
        Returns:
            DataProfile object with detailed information about the dataset
        """
        # Basic dataset metrics
        row_count = len(df)
        column_count = len(df.columns)
        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # in MB
        duplicate_rows_count = row_count - df.drop_duplicates().shape[0]
        
        # Initialize the profile
        profile = DataProfile(
            dataset_name=dataset_name,
            row_count=row_count,
            column_count=column_count,
            memory_usage=memory_usage,
            duplicate_rows_count=duplicate_rows_count
        )
        
        # Create profile for each column
        for column_name in df.columns:
            column_profile = self._create_column_profile(
                df, 
                column_name, 
                include_histograms, 
                top_n_values,
                outlier_detection_method, 
                outlier_threshold
            )
            profile.column_profiles[column_name] = column_profile
        
        # Calculate correlations if requested
        if include_correlations:
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                try:
                    corr_matrix = numeric_df.corr()
                    profile.correlations = corr_matrix.to_dict()
                except Exception as e:
                    self.logger.warning(f"Could not compute correlations: {str(e)}")
        
        return profile
    
    def _create_column_profile(self, 
                              df: pd.DataFrame, 
                              column_name: str,
                              include_histogram: bool = True,
                              top_n_values: int = 5,
                              outlier_detection_method: str = "iqr",
                              outlier_threshold: float = 1.5) -> ColumnProfile:
        """
        Create a detailed profile for a single column.
        
        Args:
            df: Pandas DataFrame containing the column
            column_name: Name of the column to profile
            include_histogram: Whether to include histogram data
            top_n_values: Number of top values to include
            outlier_detection_method: Method for outlier detection
            outlier_threshold: Threshold for outlier detection
            
        Returns:
            ColumnProfile object with detailed information about the column
        """
        column = df[column_name]
        dtype = str(column.dtype)
        count = len(column)
        missing_count = column.isna().sum()
        missing_percentage = missing_count / count if count > 0 else 0
        
        # Initialize column profile with basic information
        profile = ColumnProfile(
            name=column_name,
            dtype=dtype,
            count=count,
            missing_count=missing_count,
            missing_percentage=missing_percentage
        )
        
        # Non-missing values
        non_missing = column.dropna()
        
        # Get unique count
        try:
            profile.unique_count = non_missing.nunique()
        except:
            profile.unique_count = None
        
        # Process based on data type
        if np.issubdtype(column.dtype, np.number):
            # Numeric column
            if not non_missing.empty:
                profile.min_value = non_missing.min()
                profile.max_value = non_missing.max()
                profile.mean = non_missing.mean()
                profile.median = non_missing.median()
                profile.std_dev = non_missing.std()
                
                # Calculate quartiles and IQR for outlier detection
                profile.quartile_1 = non_missing.quantile(0.25)
                profile.quartile_3 = non_missing.quantile(0.75)
                profile.iqr = profile.quartile_3 - profile.quartile_1
                
                # Detect outliers
                profile.outliers_count = self._count_outliers(
                    non_missing, 
                    outlier_detection_method, 
                    outlier_threshold
                )
                
                # Generate histogram
                if include_histogram and len(non_missing) > 0:
                    try:
                        hist, bin_edges = np.histogram(non_missing, bins='auto')
                        profile.histogram_data = {
                            'counts': hist.tolist(),
                            'bin_edges': bin_edges.tolist()
                        }
                    except Exception as e:
                        self.logger.warning(f"Could not generate histogram for {column_name}: {str(e)}")
                
        else:
            # Categorical or other column type
            if not non_missing.empty and profile.unique_count is not None:
                # Get top values
                if profile.unique_count <= 100:  # Only for reasonably-sized categorical
                    try:
                        value_counts = non_missing.value_counts().head(top_n_values)
                        profile.top_values = list(zip(value_counts.index.tolist(), 
                                                     value_counts.tolist()))
                    except Exception as e:
                        self.logger.warning(f"Could not get top values for {column_name}: {str(e)}")
        
        return profile
    
    def _count_outliers(self, 
                      series: pd.Series, 
                      method: str = "iqr", 
                      threshold: float = 1.5) -> int:
        """
        Count outliers in a numeric series.
        
        Args:
            series: Pandas Series to analyze
            method: Method for outlier detection ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Count of outliers
        """
        if series.empty:
            return 0
        
        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            return ((series < lower_bound) | (series > upper_bound)).sum()
            
        elif method == "zscore":
            mean = series.mean()
            std = series.std()
            if std == 0:  # Avoid division by zero
                return 0
            z_scores = np.abs((series - mean) / std)
            return (z_scores > threshold).sum()
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    def _generate_summary_report(self, profile: DataProfile) -> Dict[str, Any]:
        """
        Generate a summary report from the data profile.
        
        Args:
            profile: DataProfile object with detailed information
            
        Returns:
            Dictionary with summary information
        """
        summary = {
            "dataset_name": profile.dataset_name,
            "row_count": profile.row_count,
            "column_count": profile.column_count,
            "memory_usage_mb": round(profile.memory_usage, 2),
            "duplicate_rows_percentage": round(profile.duplicate_rows_count / profile.row_count * 100 
                                              if profile.row_count > 0 else 0, 2),
            
            "column_types": {
                "numeric": len(profile.get_numeric_columns()),
                "categorical": len(profile.get_categorical_columns()),
                "datetime": len(profile.get_datetime_columns())
            },
            
            "data_quality": {
                "high_missing_columns": [
                    {
                        "name": col,
                        "missing_percentage": round(profile.column_profiles[col].missing_percentage * 100, 2)
                    }
                    for col in profile.get_high_missing_columns()
                ],
                "high_cardinality_columns": profile.get_high_cardinality_columns(),
                "columns_with_outliers": [
                    {
                        "name": col,
                        "outliers_count": profile.column_profiles[col].outliers_count,
                        "outliers_percentage": round(
                            profile.column_profiles[col].outliers_count / 
                            (profile.row_count - profile.column_profiles[col].missing_count) * 100, 2
                        ) if profile.row_count > profile.column_profiles[col].missing_count else 0
                    }
                    for col in profile.get_columns_with_outliers()
                ]
            },
            
            "potential_issues": []
        }
        
        # Add potential issues
        if summary["data_quality"]["high_missing_columns"]:
            summary["potential_issues"].append({
                "type": "missing_data",
                "description": "Some columns have high missing data percentages",
                "affected_columns": [item["name"] for item in summary["data_quality"]["high_missing_columns"]]
            })
        
        if summary["data_quality"]["columns_with_outliers"]:
            summary["potential_issues"].append({
                "type": "outliers",
                "description": "Some numeric columns contain outliers",
                "affected_columns": [item["name"] for item in summary["data_quality"]["columns_with_outliers"]]
            })
        
        if profile.duplicate_rows_count > 0:
            summary["potential_issues"].append({
                "type": "duplicate_rows",
                "description": f"Dataset contains {profile.duplicate_rows_count} duplicate rows",
                "duplicate_percentage": round(profile.duplicate_rows_count / profile.row_count * 100 
                                             if profile.row_count > 0 else 0, 2)
            })
        
        # Add correlation information if available
        if profile.correlations:
            # Find high correlations (absolute value above 0.7, but not self-correlations)
            high_correlations = []
            for col1 in profile.correlations:
                for col2 in profile.correlations[col1]:
                    if col1 != col2 and abs(profile.correlations[col1][col2]) > 0.7:
                        high_correlations.append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": round(profile.correlations[col1][col2], 2)
                        })
            
            # Add only unique pairs (avoid duplicates like A-B and B-A)
            unique_correlations = []
            for corr in high_correlations:
                reverse_pair = next(
                    (c for c in unique_correlations 
                     if c["column1"] == corr["column2"] and c["column2"] == corr["column1"]),
                    None
                )
                if not reverse_pair:
                    unique_correlations.append(corr)
            
            if unique_correlations:
                summary["high_correlations"] = unique_correlations
                summary["potential_issues"].append({
                    "type": "high_correlation",
                    "description": "Some numeric columns are highly correlated",
                    "affected_pairs": [f"{c['column1']} - {c['column2']}" for c in unique_correlations]
                })
        
        return summary
    
    def generate_correlation_matrix(self, 
                                  dataframe: pd.DataFrame, 
                                  columns: Optional[List[str]] = None,
                                  method: str = 'pearson') -> Dict[str, Any]:
        """
        Generate a correlation matrix for numeric columns in the dataframe.
        
        Args:
            dataframe: Pandas DataFrame
            columns: Specific columns to include (defaults to all numeric)
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            
        Returns:
            Dictionary with correlation matrix
        """
        self.logger.info("Generating correlation matrix")
        
        if columns:
            df_subset = dataframe[columns]
        else:
            df_subset = dataframe.select_dtypes(include=['number'])
        
        if df_subset.empty:
            return {"error": "No numeric columns available for correlation"}
        
        try:
            corr_matrix = df_subset.corr(method=method)
            
            # Convert to dictionary format
            result = {
                "method": method,
                "shape": corr_matrix.shape,
                "columns": corr_matrix.columns.tolist(),
                "data": corr_matrix.to_dict(orient='index')
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating correlation matrix: {str(e)}")
            return {"error": str(e)}
    
    def detect_outliers(self, 
                      dataframe: pd.DataFrame,
                      column: str,
                      method: str = "iqr",
                      threshold: float = 1.5) -> Dict[str, Any]:
        """
        Detect outliers in a specific column.
        
        Args:
            dataframe: Pandas DataFrame
            column: Column name to analyze
            method: Method for outlier detection ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Dictionary with outlier information
        """
        self.logger.info(f"Detecting outliers in column: {column}")
        
        if column not in dataframe.columns:
            return {"error": f"Column '{column}' not found in dataframe"}
        
        if not np.issubdtype(dataframe[column].dtype, np.number):
            return {"error": f"Column '{column}' is not numeric"}
        
        series = dataframe[column].dropna()
        
        if series.empty:
            return {"error": f"Column '{column}' has no non-missing values"}
        
        try:
            if method == "iqr":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                
                return {
                    "method": "iqr",
                    "threshold": threshold,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "outliers_count": len(outliers),
                    "outliers_percentage": len(outliers) / len(series) * 100,
                    "outliers_sample": outliers.head(10).tolist() if not outliers.empty else []
                }
                
            elif method == "zscore":
                mean = series.mean()
                std = series.std()
                if std == 0:
                    return {"error": "Standard deviation is zero, cannot compute z-scores"}
                
                z_scores = np.abs((series - mean) / std)
                outliers = series[z_scores > threshold]
                
                return {
                    "method": "zscore",
                    "threshold": threshold,
                    "mean": mean,
                    "std": std,
                    "outliers_count": len(outliers),
                    "outliers_percentage": len(outliers) / len(series) * 100,
                    "outliers_sample": outliers.head(10).tolist() if not outliers.empty else []
                }
                
            else:
                return {"error": f"Unknown outlier detection method: {method}"}
                
        except Exception as e:
            self.logger.error(f"Error detecting outliers: {str(e)}")
            return {"error": str(e)}
    
    def get_column_profile(self, 
                         dataset_name: str, 
                         column_name: str) -> Dict[str, Any]:
        """
        Get the profile for a specific column from memory.
        
        Args:
            dataset_name: Name of the dataset
            column_name: Name of the column
            
        Returns:
            Column profile information
        """
        memory_key = f"data_profile:{dataset_name}"
        profile_data = self.memory.retrieve(memory_key)
        
        if not profile_data:
            return {"error": f"No profile found for dataset: {dataset_name}"}
        
        if "column_profiles" not in profile_data or column_name not in profile_data["column_profiles"]:
            return {"error": f"No profile found for column: {column_name}"}
        
        return profile_data["column_profiles"][column_name] 