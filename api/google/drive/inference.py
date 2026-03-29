import os
import io
from typing import List, Optional
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload


# --- Google Docs export mime type mapping ---
EXPORT_MIME_TYPES = {
    "application/vnd.google-apps.document": "application/pdf",
    "application/vnd.google-apps.spreadsheet": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.google-apps.presentation": "application/pdf",
    "application/vnd.google-apps.drawing": "image/png",
}


# --- Input / Output models ---

class ListFilesInput(BaseAppInput):
    query: str = Field(default="", description="Drive search query (uses Drive query syntax)")
    max_results: int = Field(default=10, ge=1, le=1000, description="Maximum number of files to return")
    folder_id: str = Field(default="", description="Restrict listing to this folder ID")


class FileInfo(BaseAppOutput):
    id: str = Field(description="File ID")
    name: str = Field(description="File name")
    mime_type: str = Field(default="", description="MIME type")
    size: Optional[str] = Field(None, description="File size in bytes")
    modified_time: Optional[str] = Field(None, description="Last modified timestamp")
    web_view_link: Optional[str] = Field(None, description="Link to view the file in Drive")


class ListFilesOutput(BaseAppOutput):
    files: List[FileInfo] = Field(description="List of matching files")


class GetFileInput(BaseAppInput):
    file_id: str = Field(description="The ID of the file to retrieve metadata for")


class GetFileOutput(BaseAppOutput):
    id: str = Field(description="File ID")
    name: str = Field(description="File name")
    mime_type: str = Field(default="", description="MIME type")
    size: Optional[str] = Field(None, description="File size in bytes")
    modified_time: Optional[str] = Field(None, description="Last modified timestamp")
    web_view_link: Optional[str] = Field(None, description="Link to view the file in Drive")
    parents: Optional[List[str]] = Field(None, description="Parent folder IDs")


class DownloadFileInput(BaseAppInput):
    file_id: str = Field(description="The ID of the file to download")


class DownloadFileOutput(BaseAppOutput):
    file: File = Field(description="The downloaded file content")


class UploadFileInput(BaseAppInput):
    file: File = Field(description="The file to upload")
    name: str = Field(default="", description="Name for the uploaded file (defaults to original filename)")
    folder_id: str = Field(default="", description="Parent folder ID to upload into")
    mime_type: str = Field(default="", description="MIME type override for the uploaded file")


class UploadFileOutput(BaseAppOutput):
    id: str = Field(description="ID of the uploaded file")
    name: str = Field(description="Name of the uploaded file")
    web_view_link: Optional[str] = Field(None, description="Link to view the file in Drive")


class CreateFolderInput(BaseAppInput):
    name: str = Field(description="Name for the new folder")
    parent_id: str = Field(default="", description="Parent folder ID")


class CreateFolderOutput(BaseAppOutput):
    id: str = Field(description="ID of the created folder")
    name: str = Field(description="Name of the created folder")
    web_view_link: Optional[str] = Field(None, description="Link to view the folder in Drive")


class DeleteFileInput(BaseAppInput):
    file_id: str = Field(description="The ID of the file to delete")


class DeleteFileOutput(BaseAppOutput):
    success: bool = Field(description="Whether the deletion succeeded")


class SearchFilesInput(BaseAppInput):
    query: str = Field(description="Full-text search query")
    max_results: int = Field(default=10, ge=1, le=1000, description="Maximum number of results")


class SearchFilesOutput(BaseAppOutput):
    files: List[FileInfo] = Field(description="List of matching files")


# --- App ---

class App(BaseApp):
    service: object = None

    async def setup(self):
        token = os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN")
        if not token:
            raise ValueError("GOOGLE_OAUTH_ACCESS_TOKEN not found in environment")
        creds = Credentials(token=token)
        self.service = build("drive", "v3", credentials=creds)
        self.logger.info("Google Drive service initialized")

    def _file_fields(self, extra: str = "") -> str:
        base = "id, name, mimeType, size, modifiedTime, webViewLink"
        if extra:
            base += f", {extra}"
        return base

    def _to_file_info(self, f: dict) -> FileInfo:
        return FileInfo(
            id=f["id"],
            name=f["name"],
            mime_type=f.get("mimeType", ""),
            size=f.get("size"),
            modified_time=f.get("modifiedTime"),
            web_view_link=f.get("webViewLink"),
        )

    # --- list_files ---
    async def list_files(self, input_data: ListFilesInput) -> ListFilesOutput:
        self.logger.info(f"Listing files: query={input_data.query!r}, folder_id={input_data.folder_id!r}, max={input_data.max_results}")

        q_parts = []
        if input_data.folder_id:
            q_parts.append(f"'{input_data.folder_id}' in parents")
        if input_data.query:
            q_parts.append(input_data.query)
        q = " and ".join(q_parts) if q_parts else None

        results = (
            self.service.files()
            .list(
                q=q,
                pageSize=input_data.max_results,
                fields=f"files({self._file_fields()})",
            )
            .execute()
        )

        files = [self._to_file_info(f) for f in results.get("files", [])]
        self.logger.info(f"Found {len(files)} files")
        return ListFilesOutput(files=files)

    # --- get_file ---
    async def get_file(self, input_data: GetFileInput) -> GetFileOutput:
        self.logger.info(f"Getting file metadata: {input_data.file_id}")

        f = (
            self.service.files()
            .get(fileId=input_data.file_id, fields=self._file_fields("parents"))
            .execute()
        )

        return GetFileOutput(
            id=f["id"],
            name=f["name"],
            mime_type=f.get("mimeType", ""),
            size=f.get("size"),
            modified_time=f.get("modifiedTime"),
            web_view_link=f.get("webViewLink"),
            parents=f.get("parents"),
        )

    # --- download_file ---
    async def download_file(self, input_data: DownloadFileInput) -> DownloadFileOutput:
        self.logger.info(f"Downloading file: {input_data.file_id}")

        # Get file metadata first to determine name and type
        meta = (
            self.service.files()
            .get(fileId=input_data.file_id, fields="id, name, mimeType")
            .execute()
        )
        file_name = meta["name"]
        mime_type = meta.get("mimeType", "")

        output_path = f"/tmp/{file_name}"

        # Google Workspace files need export, regular files use get_media
        if mime_type in EXPORT_MIME_TYPES:
            export_mime = EXPORT_MIME_TYPES[mime_type]
            self.logger.info(f"Exporting Google Workspace file as {export_mime}")
            request = self.service.files().export_media(
                fileId=input_data.file_id, mimeType=export_mime
            )
            # Adjust extension for exported files
            ext_map = {
                "application/pdf": ".pdf",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
                "image/png": ".png",
            }
            ext = ext_map.get(export_mime, "")
            if ext and not output_path.endswith(ext):
                output_path += ext
        else:
            request = self.service.files().get_media(fileId=input_data.file_id)

        fh = io.FileIO(output_path, "wb")
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.close()

        self.logger.info(f"File downloaded to {output_path}")
        return DownloadFileOutput(file=File(path=output_path))

    # --- upload_file ---
    async def upload_file(self, input_data: UploadFileInput) -> UploadFileOutput:
        file_name = input_data.name or os.path.basename(input_data.file.path)
        self.logger.info(f"Uploading file: {file_name}")

        file_metadata = {"name": file_name}
        if input_data.folder_id:
            file_metadata["parents"] = [input_data.folder_id]

        mime_type = input_data.mime_type or input_data.file.content_type or "application/octet-stream"
        media = MediaFileUpload(input_data.file.path, mimetype=mime_type, resumable=True)

        result = (
            self.service.files()
            .create(body=file_metadata, media_body=media, fields="id, name, webViewLink")
            .execute()
        )

        self.logger.info(f"Uploaded file: id={result['id']}, name={result['name']}")
        return UploadFileOutput(
            id=result["id"],
            name=result["name"],
            web_view_link=result.get("webViewLink"),
        )

    # --- create_folder ---
    async def create_folder(self, input_data: CreateFolderInput) -> CreateFolderOutput:
        self.logger.info(f"Creating folder: {input_data.name}")

        file_metadata = {
            "name": input_data.name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if input_data.parent_id:
            file_metadata["parents"] = [input_data.parent_id]

        result = (
            self.service.files()
            .create(body=file_metadata, fields="id, name, webViewLink")
            .execute()
        )

        self.logger.info(f"Created folder: id={result['id']}, name={result['name']}")
        return CreateFolderOutput(
            id=result["id"],
            name=result["name"],
            web_view_link=result.get("webViewLink"),
        )

    # --- delete_file ---
    async def delete_file(self, input_data: DeleteFileInput) -> DeleteFileOutput:
        self.logger.info(f"Deleting file: {input_data.file_id}")

        self.service.files().delete(fileId=input_data.file_id).execute()

        self.logger.info(f"Deleted file: {input_data.file_id}")
        return DeleteFileOutput(success=True)

    # --- search_files ---
    async def search_files(self, input_data: SearchFilesInput) -> SearchFilesOutput:
        self.logger.info(f"Searching files: query={input_data.query!r}, max={input_data.max_results}")

        q = f"fullText contains '{input_data.query}'"
        results = (
            self.service.files()
            .list(
                q=q,
                pageSize=input_data.max_results,
                fields=f"files({self._file_fields()})",
            )
            .execute()
        )

        files = [self._to_file_info(f) for f in results.get("files", [])]
        self.logger.info(f"Search returned {len(files)} files")
        return SearchFilesOutput(files=files)
