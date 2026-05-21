import os
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

class AppInput(BaseAppInput):
    service: str = Field(default="drive", enum=["drive", "sheets", "docs", "gmail", "calendar"], description="Google service to test: drive, sheets, docs, gmail, calendar")

class AppOutput(BaseAppOutput):
    success: bool = Field(description="Whether the operation was successful")
    token_exists: bool = Field(description="Whether the required access token exists in environment")
    response: str = Field(default="", description="Response from Google API")
    error: str = Field(default="", description="Error message if any")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize your model and resources here."""
        self.access_token = os.environ.get("GOOGLE_ACCESS_TOKEN")
        self.oauth_access_token = os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run prediction on the input data."""
        service_name = input_data.service.lower()
        
        # Gmail and Calendar use GOOGLE_OAUTH_ACCESS_TOKEN
        if service_name in ["gmail", "calendar"]:
            token = self.oauth_access_token
            token_env_var = "GOOGLE_OAUTH_ACCESS_TOKEN"
        else:
            token = self.access_token
            token_env_var = "GOOGLE_ACCESS_TOKEN"
        
        token_exists = token is not None
        
        if not token_exists:
            return AppOutput(
                success=False,
                token_exists=False,
                error=f"{token_env_var} not found in environment"
            )
        
        try:
            # Create credentials from the access token
            creds = Credentials(token=token)
            
            if service_name == "drive":
                service = build("drive", "v3", credentials=creds)
                results = service.files().list(pageSize=5, fields="files(id, name)").execute()
                files = results.get("files", [])
                response = f"Found {len(files)} files: {[f['name'] for f in files]}"
                
            elif service_name == "sheets":
                service = build("sheets", "v4", credentials=creds)
                # Just verify we can build the service
                response = "Sheets API client initialized successfully"
                
            elif service_name == "docs":
                service = build("docs", "v1", credentials=creds)
                response = "Docs API client initialized successfully"
                
            elif service_name == "gmail":
                service = build("gmail", "v1", credentials=creds)
                results = service.users().labels().list(userId="me").execute()
                labels = results.get("labels", [])
                response = f"Found {len(labels)} Gmail labels"
                
            elif service_name == "calendar":
                service = build("calendar", "v3", credentials=creds)
                results = service.calendarList().list(maxResults=10).execute()
                calendars = results.get("items", [])
                response = f"Found {len(calendars)} calendars: {[c.get('summary', 'Unknown') for c in calendars]}"
                
            else:
                return AppOutput(
                    success=False,
                    token_exists=True,
                    error=f"Unknown service: {service_name}. Use: drive, sheets, docs, gmail, calendar"
                )
            
            return AppOutput(
                success=True,
                token_exists=True,
                response=response
            )
            
        except Exception as e:
            return AppOutput(
                success=False,
                token_exists=True,
                error=str(e)
            )
