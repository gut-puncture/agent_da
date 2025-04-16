"""
Google Sheets Data Connector
--------------------------
Handles authentication and data retrieval from Google Sheets.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import re
import tempfile

from agent_framework import BaseAgent, AgentMemory, AgentState, DatasetInfo

logger = logging.getLogger("DataConnector")

class GoogleSheetsConnector(BaseAgent):
    """Agent for connecting to Google Sheets and retrieving data."""
    
    # If modifying these scopes, delete the file token.json.
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly',
              'https://www.googleapis.com/auth/spreadsheets.readonly']
    
    def __init__(self, agent_id: str, memory: AgentMemory, 
                 credentials_path: Optional[str] = None,
                 token_path: Optional[str] = 'token.json'):
        super().__init__(agent_id, memory)
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.credentials = None
        self.drive_service = None
        self.sheets_service = None
        
    def _execute(self, action: str = "authenticate", **kwargs) -> Any:
        """Main execution method for the Google Sheets connector."""
        if action == "authenticate":
            return self._authenticate()
        elif action == "list_files":
            return self.list_files(**kwargs)
        elif action == "get_sheet_data":
            return self.get_sheet_data(**kwargs)
        elif action == "search_files":
            return self.search_files(**kwargs)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def _authenticate(self) -> bool:
        """Authenticate with Google API."""
        try:
            credentials = None
            # Try to load saved credentials
            if os.path.exists(self.token_path):
                try:
                    credentials = Credentials.from_authorized_user_info(
                        json.load(open(self.token_path)), self.SCOPES)
                except Exception as e:
                    logger.warning(f"Error loading credentials from token file: {str(e)}")
            
            # Check if credentials are valid
            if credentials and credentials.valid:
                if credentials.expired and credentials.refresh_token:
                    credentials.refresh(Request())
            else:
                # Use service account if provided
                if self.credentials_path and os.path.exists(self.credentials_path):
                    if self.credentials_path.endswith('.json'):
                        try:
                            # Check if it's a service account
                            cred_data = json.load(open(self.credentials_path))
                            if 'type' in cred_data and cred_data['type'] == 'service_account':
                                credentials = service_account.Credentials.from_service_account_file(
                                    self.credentials_path, scopes=self.SCOPES)
                            else:
                                # It's OAuth client credentials
                                flow = InstalledAppFlow.from_client_secrets_file(
                                    self.credentials_path, self.SCOPES)
                                credentials = flow.run_local_server(port=0)
                        except Exception as e:
                            logger.error(f"Error loading credentials: {str(e)}")
                            raise
                else:
                    raise ValueError("No valid credentials found. Please provide a credentials file.")
            
            # Save the credentials for the next run
            if credentials and hasattr(credentials, 'to_json'):
                with open(self.token_path, 'w') as token:
                    token.write(credentials.to_json())
            
            self.credentials = credentials
            self.drive_service = build('drive', 'v3', credentials=credentials)
            self.sheets_service = build('sheets', 'v4', credentials=credentials)
            
            logger.info("Successfully authenticated with Google API")
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            self.state = AgentState.FAILED
            raise
    
    def list_files(self, file_type: str = 'spreadsheet', limit: int = 10) -> List[Dict[str, Any]]:
        """List available files of the specified type."""
        if not self.drive_service:
            self._authenticate()
        
        query = ""
        if file_type == 'spreadsheet':
            query = "mimeType='application/vnd.google-apps.spreadsheet'"
        
        try:
            results = self.drive_service.files().list(
                q=query,
                pageSize=limit,
                fields="nextPageToken, files(id, name, createdTime, modifiedTime, webViewLink)"
            ).execute()
            
            files = results.get('files', [])
            if not files:
                logger.info('No files found.')
                return []
            
            # Store files in memory for future reference
            self.memory.store('google_files', files)
            
            return files
            
        except HttpError as error:
            logger.error(f'An error occurred: {error}')
            return []
    
    def search_files(self, query: str, file_type: str = 'spreadsheet', limit: int = 10) -> List[Dict[str, Any]]:
        """Search for files matching the query."""
        if not self.drive_service:
            self._authenticate()
        
        search_query = f"name contains '{query}'"
        if file_type == 'spreadsheet':
            search_query += " and mimeType='application/vnd.google-apps.spreadsheet'"
        
        try:
            results = self.drive_service.files().list(
                q=search_query,
                pageSize=limit,
                fields="nextPageToken, files(id, name, createdTime, modifiedTime, webViewLink)"
            ).execute()
            
            files = results.get('files', [])
            return files
            
        except HttpError as error:
            logger.error(f'An error occurred: {error}')
            return []
    
    def get_sheet_names(self, spreadsheet_id: str) -> List[str]:
        """Get the names of all sheets in a spreadsheet."""
        if not self.sheets_service:
            self._authenticate()
        
        try:
            spreadsheet = self.sheets_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
            sheets = spreadsheet.get('sheets', [])
            return [sheet['properties']['title'] for sheet in sheets]
        except HttpError as error:
            logger.error(f'An error occurred: {error}')
            return []
    
    def get_sheet_data(self, spreadsheet_id: str, sheet_name: Optional[str] = None, 
                      range_name: Optional[str] = None, header_row: int = 0,
                      infer_types: bool = True) -> Tuple[pd.DataFrame, DatasetInfo]:
        """
        Retrieve data from a specified Google Sheet and convert to pandas DataFrame.
        
        Args:
            spreadsheet_id: The ID of the spreadsheet
            sheet_name: Name of the sheet to retrieve (if None, uses the first sheet)
            range_name: Range to retrieve (e.g., 'A1:Z100'). If None, retrieves all data.
            header_row: Row number to use as column headers (0-based)
            infer_types: Whether to attempt to convert columns to appropriate data types
            
        Returns:
            Tuple of (DataFrame with the sheet data, DatasetInfo object)
        """
        if not self.sheets_service:
            self._authenticate()
        
        try:
            # Get sheet names if not provided
            if not sheet_name:
                sheet_names = self.get_sheet_names(spreadsheet_id)
                if not sheet_names:
                    raise ValueError("No sheets found in the spreadsheet")
                sheet_name = sheet_names[0]
            
            # Construct the range
            sheet_range = f"'{sheet_name}'"
            if range_name:
                sheet_range += f"!{range_name}"
            
            # Get the data
            result = self.sheets_service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=sheet_range,
                valueRenderOption='UNFORMATTED_VALUE'
            ).execute()
            
            values = result.get('values', [])
            if not values:
                logger.warning(f'No data found in {sheet_name}')
                return pd.DataFrame(), DatasetInfo(
                    name=sheet_name,
                    source=f"sheets:{spreadsheet_id}:{sheet_name}",
                    rows=0,
                    columns=0,
                    column_types={}
                )
            
            # Extract headers and data
            headers = values[header_row] if len(values) > header_row else []
            data_rows = values[header_row + 1:] if len(values) > header_row + 1 else []
            
            # Clean up headers - replace empty or duplicate headers
            cleaned_headers = []
            seen_headers = set()
            for i, header in enumerate(headers):
                if not header or header in seen_headers:
                    header = f"column_{i+1}"
                # Ensure header is string
                header = str(header)
                # Add suffix for duplicates
                original_header = header
                suffix = 1
                while header in seen_headers:
                    header = f"{original_header}_{suffix}"
                    suffix += 1
                cleaned_headers.append(header)
                seen_headers.add(header)
            
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=cleaned_headers)
            
            # Handle empty cells
            df = df.replace('', np.nan)
            
            # Infer types for columns
            if infer_types:
                for col in df.columns:
                    try:
                        # Try to convert to numeric
                        numeric_values = pd.to_numeric(df[col], errors='coerce')
                        # If more than 70% converted successfully, use numeric
                        if numeric_values.notna().sum() / len(numeric_values) > 0.7:
                            df[col] = numeric_values
                            continue
                        
                        # Try to convert to datetime
                        datetime_values = pd.to_datetime(df[col], errors='coerce')
                        # If more than 70% converted successfully, use datetime
                        if datetime_values.notna().sum() / len(datetime_values) > 0.7:
                            df[col] = datetime_values
                    except Exception as e:
                        logger.debug(f"Error converting column {col}: {str(e)}")
            
            # Create dataset info
            dataset_info = DatasetInfo.from_dataframe(
                name=sheet_name,
                source=f"sheets:{spreadsheet_id}:{sheet_name}",
                df=df
            )
            
            # Store the data in memory
            dataset_key = f"sheets_data:{spreadsheet_id}:{sheet_name}"
            self.memory.store(dataset_key, df)
            self.memory.store(f"{dataset_key}:info", dataset_info)
            
            logger.info(f"Successfully retrieved sheet data: {len(df)} rows, {len(df.columns)} columns")
            return df, dataset_info
            
        except HttpError as error:
            logger.error(f'An error occurred: {error}')
            raise
    
    def export_sheet_to_csv(self, spreadsheet_id: str, sheet_name: Optional[str] = None) -> str:
        """Export a Google Sheet to a CSV file and return the file path."""
        df, _ = self.get_sheet_data(spreadsheet_id, sheet_name)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp:
            temp_path = temp.name
        
        # Save the DataFrame to CSV
        df.to_csv(temp_path, index=False)
        return temp_path

class MultipleSheetConnector(BaseAgent):
    """Agent for handling multiple Google Sheets and combining data."""
    
    def __init__(self, agent_id: str, memory: AgentMemory, google_connector: GoogleSheetsConnector):
        super().__init__(agent_id, memory)
        self.google_connector = google_connector
    
    def _execute(self, action: str = "combine_sheets", **kwargs) -> Any:
        """Main execution method for the multiple sheet connector."""
        if action == "combine_sheets":
            return self.combine_sheets(**kwargs)
        elif action == "join_sheets":
            return self.join_sheets(**kwargs)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def combine_sheets(self, spreadsheet_ids: List[str], sheet_names: Optional[List[str]] = None,
                      prefix_source: bool = True) -> Tuple[Dict[str, pd.DataFrame], Dict[str, DatasetInfo]]:
        """
        Retrieve data from multiple sheets and return as a dictionary of DataFrames.
        
        Args:
            spreadsheet_ids: List of spreadsheet IDs
            sheet_names: Optional list of sheet names (must be same length as spreadsheet_ids)
            prefix_source: Whether to prefix column names with the source
            
        Returns:
            Dictionary mapping sheet identifiers to DataFrames and corresponding DatasetInfo objects.
        """
        if sheet_names and len(spreadsheet_ids) != len(sheet_names):
            raise ValueError("If sheet_names is provided, it must have the same length as spreadsheet_ids")
        
        # Use None for sheet_names if not provided
        if not sheet_names:
            sheet_names = [None] * len(spreadsheet_ids)
        
        datasets = {}
        dataset_infos = {}
        
        for i, (spreadsheet_id, sheet_name) in enumerate(zip(spreadsheet_ids, sheet_names)):
            try:
                df, info = self.google_connector.get_sheet_data(spreadsheet_id, sheet_name)
                
                # Add prefix to columns if requested
                if prefix_source:
                    prefix = f"sheet{i+1}_"
                    df.columns = [f"{prefix}{col}" for col in df.columns]
                
                sheet_id = sheet_name if sheet_name else f"sheet_{i+1}"
                key = f"{spreadsheet_id}:{sheet_id}"
                datasets[key] = df
                dataset_infos[key] = info
                
            except Exception as e:
                logger.error(f"Error retrieving sheet {spreadsheet_id}:{sheet_name}: {str(e)}")
        
        # Store combined data in memory
        self.memory.store("combined_sheets", datasets)
        self.memory.store("combined_sheets_info", dataset_infos)
        
        return datasets, dataset_infos
    
    def join_sheets(self, spreadsheet_ids: List[str], sheet_names: Optional[List[str]] = None,
                    join_columns: List[str] = None, join_type: str = 'inner') -> pd.DataFrame:
        """
        Join multiple sheets on specified columns.
        
        Args:
            spreadsheet_ids: List of spreadsheet IDs.
            sheet_names: Optional list of sheet names.
            join_columns: List of column names to join on (one per sheet).
            join_type: Type of join ('inner', 'outer', 'left', 'right').
            
        Returns:
            DataFrame with joined data.
        """
        datasets, _ = self.combine_sheets(spreadsheet_ids, sheet_names, prefix_source=False)
        
        if len(datasets) < 2:
            raise ValueError("Need at least two sheets to join")
        
        # Get the DataFrames in order
        dfs = list(datasets.values())
        
        # If join columns not specified, try to find common columns
        if not join_columns:
            # Get the intersection of all column names
            common_cols = set(dfs[0].columns)
            for df in dfs[1:]:
                common_cols &= set(df.columns)
            
            if not common_cols:
                raise ValueError("No common columns found for joining and no join_columns specified")
            
            # Use the first common column for all sheets
            join_columns = [list(common_cols)[0]] * len(dfs)
        
        if len(join_columns) != len(dfs):
            # If only one join column provided, use it for all dataframes
            if len(join_columns) == 1:
                join_columns = join_columns * len(dfs)
            else:
                raise ValueError("Number of join_columns must match number of sheets or be just one column")
        
        # Rename the join column in each dataframe to a common key for merging
        common_join = "join_key"
        renamed_dfs = []
        for i, df in enumerate(dfs):
            join_col = join_columns[i]
            if join_col not in df.columns:
                raise ValueError(f"Join column {join_col} not found in sheet {i+1}")
            renamed_df = df.rename(columns={join_col: common_join})
            renamed_dfs.append(renamed_df)
        
        # Iteratively merge all dataframes on the common join key
        merged_df = renamed_dfs[0]
        for df in renamed_dfs[1:]:
            merged_df = pd.merge(merged_df, df, on=common_join, how=join_type)
        
        return merged_df
