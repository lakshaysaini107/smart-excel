import pandas as pd
from pathlib import Path

class ExcelHandler:
    """Handles Excel file reading and processing"""
    
    @staticmethod
    def read_excel_file(file_path):
        """Read Excel file and return as ExcelFile object"""
        try:
            excel_file = pd.ExcelFile(file_path)
            return excel_file
        except Exception as e:
            raise Exception(f"Error reading Excel file: {str(e)}")
    
    @staticmethod
    def get_sheet_names(file_path):
        """Get all sheet names from Excel file"""
        try:
            excel_file = ExcelHandler.read_excel_file(file_path)
            return excel_file.sheet_names
        except Exception as e:
            raise Exception(f"Error getting sheet names: {str(e)}")
    
    @staticmethod
    def get_dataframe_from_sheet(file_path, sheet_name):
        """Load specific sheet as DataFrame"""
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            return df
        except Exception as e:
            raise Exception(f"Error reading sheet: {str(e)}")
    
    @staticmethod
    def get_dataframe_description(df):
        """Get detailed description of DataFrame for LLM"""
        description = {
            "shape": {
                "rows": df.shape[0],
                "columns": df.shape[1]
            },
            "columns": df.columns.tolist(),
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_data": df.head(3).to_dict('records')
        }
        return description
