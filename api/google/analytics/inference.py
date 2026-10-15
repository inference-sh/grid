import os
import logging
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, function
from pydantic import Field
from typing import Dict, List, Optional
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True}
CREDENTIALS = [{"key": "google.analytics.readonly"}]


# --- list_properties ---

class ListPropertiesInput(BaseAppInput):
    pass


class Property(BaseAppOutput):
    property_id: str = Field(description="Property ID, e.g. 123456789; pass it to run_report")
    display_name: str = Field(description="Property name")
    account_id: str = Field(description="Account ID")
    account_name: str = Field(description="Account name")


class ListPropertiesOutput(BaseAppOutput):
    properties: List[Property] = Field(description="GA4 properties you can access")


# --- run_report ---

class DateRange(BaseAppInput):
    start_date: str = Field(description="YYYY-MM-DD, or relative: today, yesterday, 7daysAgo, 28daysAgo")
    end_date: str = Field(description="YYYY-MM-DD, or relative: today, yesterday")


class RunReportInput(BaseAppInput):
    property_id: str = Field(description="Property ID from list_properties, e.g. 123456789")
    metrics: List[str] = Field(default=["activeUsers", "sessions"], description="GA4 metrics, e.g. activeUsers, sessions, screenPageViews, conversions, engagementRate, totalRevenue")
    dimensions: List[str] = Field(default=[], description="GA4 dimensions, e.g. date, pagePath, sessionSource, sessionMedium, sessionDefaultChannelGroup, country, deviceCategory")
    date_ranges: List[DateRange] = Field(default=[DateRange(start_date="28daysAgo", end_date="yesterday")], description="Date ranges to report on")
    order_by_metric: Optional[str] = Field(default=None, description="Sort descending by this metric")
    limit: int = Field(default=100, description="Maximum rows, up to 250000")
    offset: int = Field(default=0, description="Row offset for paging")


class ReportRow(BaseAppOutput):
    dimensions: Dict[str, str] = Field(description="Dimension name to value")
    metrics: Dict[str, str] = Field(description="Metric name to value")


class RunReportOutput(BaseAppOutput):
    rows: List[ReportRow] = Field(description="Report rows")
    row_count: int = Field(description="Total rows the query matched, before limit")


# --- run_realtime_report ---

class RunRealtimeReportInput(BaseAppInput):
    property_id: str = Field(description="Property ID from list_properties")
    metrics: List[str] = Field(default=["activeUsers"], description="Realtime metrics, e.g. activeUsers, screenPageViews, eventCount")
    dimensions: List[str] = Field(default=[], description="Realtime dimensions, e.g. unifiedScreenName, country, deviceCategory")
    limit: int = Field(default=100, description="Maximum rows")


def _creds() -> Credentials:
    token = os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN")
    if not token:
        raise ValueError(
            "GOOGLE_OAUTH_ACCESS_TOKEN not found. "
            "Please ensure the Google Analytics integration is connected in Settings."
        )
    return Credentials(token=token)


def _property(property_id: str) -> str:
    return property_id if property_id.startswith("properties/") else f"properties/{property_id}"


def _rows(result: dict) -> List[ReportRow]:
    dim_names = [h["name"] for h in result.get("dimensionHeaders", [])]
    metric_names = [h["name"] for h in result.get("metricHeaders", [])]
    return [
        ReportRow(
            dimensions={n: v.get("value", "") for n, v in zip(dim_names, r.get("dimensionValues", []))},
            metrics={n: v.get("value", "") for n, v in zip(metric_names, r.get("metricValues", []))},
        )
        for r in result.get("rows", [])
    ]


class App(BaseApp):

    async def setup(self):
        self.logger = logging.getLogger(__name__)

    @function(credentials=CREDENTIALS, annotations={"title": "List properties", **READ_ONLY})
    async def list_properties(self, input_data: ListPropertiesInput) -> ListPropertiesOutput:
        """List the GA4 properties you can access, with their accounts."""
        self.logger.info("Listing GA4 properties")
        service = build("analyticsadmin", "v1beta", credentials=_creds())
        properties = []
        page_token = None
        while True:
            result = service.accountSummaries().list(pageSize=200, pageToken=page_token).execute()
            for account in result.get("accountSummaries", []):
                for p in account.get("propertySummaries", []):
                    properties.append(Property(
                        property_id=p["property"].split("/")[-1],
                        display_name=p.get("displayName", ""),
                        account_id=account.get("account", "").split("/")[-1],
                        account_name=account.get("displayName", ""),
                    ))
            page_token = result.get("nextPageToken")
            if not page_token:
                break
        self.logger.info(f"Found {len(properties)} properties")
        return ListPropertiesOutput(properties=properties)

    @function(credentials=CREDENTIALS, annotations={"title": "Run report", **READ_ONLY})
    async def run_report(self, input_data: RunReportInput) -> RunReportOutput:
        """Run a GA4 report: metrics such as users, sessions and conversions, grouped by dimensions such as date, page or source."""
        body = {
            "metrics": [{"name": m} for m in input_data.metrics],
            "dimensions": [{"name": d} for d in input_data.dimensions],
            "dateRanges": [{"startDate": r.start_date, "endDate": r.end_date} for r in input_data.date_ranges],
            "limit": input_data.limit,
            "offset": input_data.offset,
        }
        if input_data.order_by_metric:
            body["orderBys"] = [{"metric": {"metricName": input_data.order_by_metric}, "desc": True}]
        self.logger.info(f"Running report on {input_data.property_id}: {input_data.metrics} by {input_data.dimensions}")
        service = build("analyticsdata", "v1beta", credentials=_creds())
        result = service.properties().runReport(property=_property(input_data.property_id), body=body).execute()
        rows = _rows(result)
        self.logger.info(f"Got {len(rows)} rows")
        return RunReportOutput(rows=rows, row_count=result.get("rowCount", len(rows)))

    @function(credentials=CREDENTIALS, annotations={"title": "Run realtime report", **READ_ONLY})
    async def run_realtime_report(self, input_data: RunRealtimeReportInput) -> RunReportOutput:
        """Activity on the property in the last 30 minutes."""
        body = {
            "metrics": [{"name": m} for m in input_data.metrics],
            "dimensions": [{"name": d} for d in input_data.dimensions],
            "limit": input_data.limit,
        }
        self.logger.info(f"Running realtime report on {input_data.property_id}")
        service = build("analyticsdata", "v1beta", credentials=_creds())
        result = service.properties().runRealtimeReport(property=_property(input_data.property_id), body=body).execute()
        rows = _rows(result)
        return RunReportOutput(rows=rows, row_count=result.get("rowCount", len(rows)))
