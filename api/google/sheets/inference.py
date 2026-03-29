import os
import logging
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import List, Optional
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


# --- read_sheet ---

class ReadSheetInput(BaseAppInput):
    spreadsheet_id: str = Field(description="The ID of the spreadsheet to read")
    range: str = Field(default="Sheet1", description="A1 notation range to read (e.g. 'Sheet1!A1:D10')")


class ReadSheetOutput(BaseAppOutput):
    values: List[List[str]] = Field(description="2D array of cell values")
    rows: int = Field(description="Number of rows returned")
    cols: int = Field(description="Number of columns returned")


# --- write_sheet ---

class WriteSheetInput(BaseAppInput):
    spreadsheet_id: str = Field(description="The ID of the spreadsheet to write to")
    range: str = Field(default="Sheet1", description="A1 notation range to write (e.g. 'Sheet1!A1')")
    values: List[List[str]] = Field(default=[[]], description="2D array of values to write")


class WriteSheetOutput(BaseAppOutput):
    updated_range: str = Field(description="The range that was updated")
    updated_rows: int = Field(description="Number of rows updated")
    updated_cols: int = Field(description="Number of columns updated")


# --- append_rows ---

class AppendRowsInput(BaseAppInput):
    spreadsheet_id: str = Field(description="The ID of the spreadsheet to append to")
    range: str = Field(default="Sheet1", description="A1 notation range to append after (e.g. 'Sheet1')")
    values: List[List[str]] = Field(default=[[]], description="2D array of rows to append")


class AppendRowsOutput(BaseAppOutput):
    updated_range: str = Field(description="The range that was appended to")
    updated_rows: int = Field(description="Number of rows appended")


# --- create_spreadsheet ---

class CreateSpreadsheetInput(BaseAppInput):
    title: str = Field(description="Title for the new spreadsheet")


class CreateSpreadsheetOutput(BaseAppOutput):
    id: str = Field(description="The new spreadsheet ID")
    url: str = Field(description="URL to open the spreadsheet")


# --- get_spreadsheet ---

class SheetInfo(BaseAppOutput):
    id: int = Field(description="Sheet ID within the spreadsheet")
    title: str = Field(description="Sheet tab title")
    row_count: int = Field(description="Number of rows in the sheet")
    col_count: int = Field(description="Number of columns in the sheet")


class GetSpreadsheetInput(BaseAppInput):
    spreadsheet_id: str = Field(description="The ID of the spreadsheet to retrieve")


class GetSpreadsheetOutput(BaseAppOutput):
    id: str = Field(description="Spreadsheet ID")
    title: str = Field(description="Spreadsheet title")
    url: str = Field(description="URL to open the spreadsheet")
    sheets: List[SheetInfo] = Field(description="List of sheets/tabs in the spreadsheet")


# --- clear_range ---

class ClearRangeInput(BaseAppInput):
    spreadsheet_id: str = Field(description="The ID of the spreadsheet")
    range: str = Field(description="A1 notation range to clear (e.g. 'Sheet1!A1:D10')")


class ClearRangeOutput(BaseAppOutput):
    cleared_range: str = Field(description="The range that was cleared")


# --- App ---

class App(BaseApp):

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        token = os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN")
        if not token:
            raise ValueError("GOOGLE_OAUTH_ACCESS_TOKEN not found")
        creds = Credentials(token=token)
        self.service = build("sheets", "v4", credentials=creds)

    async def read_sheet(self, input_data: ReadSheetInput) -> ReadSheetOutput:
        self.logger.info(f"Reading spreadsheet {input_data.spreadsheet_id} range={input_data.range}")
        result = self.service.spreadsheets().values().get(
            spreadsheetId=input_data.spreadsheet_id,
            range=input_data.range,
        ).execute()
        values = result.get("values", [])
        rows = len(values)
        cols = max((len(row) for row in values), default=0)
        self.logger.info(f"Read {rows} rows, {cols} cols")
        return ReadSheetOutput(values=values, rows=rows, cols=cols)

    async def write_sheet(self, input_data: WriteSheetInput) -> WriteSheetOutput:
        self.logger.info(f"Writing to spreadsheet {input_data.spreadsheet_id} range={input_data.range}")
        body = {"values": input_data.values}
        result = self.service.spreadsheets().values().update(
            spreadsheetId=input_data.spreadsheet_id,
            range=input_data.range,
            valueInputOption="USER_ENTERED",
            body=body,
        ).execute()
        self.logger.info(f"Updated {result.get('updatedRows', 0)} rows, {result.get('updatedColumns', 0)} cols")
        return WriteSheetOutput(
            updated_range=result.get("updatedRange", ""),
            updated_rows=result.get("updatedRows", 0),
            updated_cols=result.get("updatedColumns", 0),
        )

    async def append_rows(self, input_data: AppendRowsInput) -> AppendRowsOutput:
        self.logger.info(f"Appending to spreadsheet {input_data.spreadsheet_id} range={input_data.range}")
        body = {"values": input_data.values}
        result = self.service.spreadsheets().values().append(
            spreadsheetId=input_data.spreadsheet_id,
            range=input_data.range,
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body=body,
        ).execute()
        updates = result.get("updates", {})
        self.logger.info(f"Appended {updates.get('updatedRows', 0)} rows")
        return AppendRowsOutput(
            updated_range=updates.get("updatedRange", ""),
            updated_rows=updates.get("updatedRows", 0),
        )

    async def create_spreadsheet(self, input_data: CreateSpreadsheetInput) -> CreateSpreadsheetOutput:
        self.logger.info(f"Creating spreadsheet: {input_data.title}")
        body = {"properties": {"title": input_data.title}}
        result = self.service.spreadsheets().create(body=body).execute()
        spreadsheet_id = result["spreadsheetId"]
        url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit"
        self.logger.info(f"Created spreadsheet {spreadsheet_id}")
        return CreateSpreadsheetOutput(id=spreadsheet_id, url=url)

    async def get_spreadsheet(self, input_data: GetSpreadsheetInput) -> GetSpreadsheetOutput:
        self.logger.info(f"Getting spreadsheet metadata: {input_data.spreadsheet_id}")
        result = self.service.spreadsheets().get(
            spreadsheetId=input_data.spreadsheet_id,
        ).execute()
        sheets = []
        for s in result.get("sheets", []):
            props = s.get("properties", {})
            grid = props.get("gridProperties", {})
            sheets.append(SheetInfo(
                id=props.get("sheetId", 0),
                title=props.get("title", ""),
                row_count=grid.get("rowCount", 0),
                col_count=grid.get("columnCount", 0),
            ))
        url = f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet_id}/edit"
        self.logger.info(f"Spreadsheet has {len(sheets)} sheets")
        return GetSpreadsheetOutput(
            id=result["spreadsheetId"],
            title=result.get("properties", {}).get("title", ""),
            url=url,
            sheets=sheets,
        )

    async def clear_range(self, input_data: ClearRangeInput) -> ClearRangeOutput:
        self.logger.info(f"Clearing range {input_data.range} in {input_data.spreadsheet_id}")
        result = self.service.spreadsheets().values().clear(
            spreadsheetId=input_data.spreadsheet_id,
            range=input_data.range,
            body={},
        ).execute()
        self.logger.info(f"Cleared range: {result.get('clearedRange', '')}")
        return ClearRangeOutput(cleared_range=result.get("clearedRange", ""))
