import os
import logging
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, function
from pydantic import Field
from typing import List, Optional, Any
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True}
CREDENTIALS = [{"key": "google.searchconsole.readonly"}]


# --- list_sites ---

class ListSitesInput(BaseAppInput):
    pass


class Site(BaseAppOutput):
    site_url: str = Field(description="Property URL, e.g. https://example.com/ or sc-domain:example.com")
    permission_level: str = Field(description="Your access: siteOwner, siteFullUser, siteRestrictedUser or siteUnverifiedUser")


class ListSitesOutput(BaseAppOutput):
    sites: List[Site] = Field(description="Search Console properties you can access")


# --- query_search_analytics ---

class QuerySearchAnalyticsInput(BaseAppInput):
    site_url: str = Field(description="Property URL exactly as list_sites returns it, e.g. https://example.com/ or sc-domain:example.com")
    start_date: str = Field(description="Start date, YYYY-MM-DD")
    end_date: str = Field(description="End date, YYYY-MM-DD")
    dimensions: List[str] = Field(default=["query"], description="Group by any of: query, page, country, device, searchAppearance, date")
    search_type: str = Field(default="web", description="web, image, video, news, discover or googleNews")
    query_contains: Optional[str] = Field(default=None, description="Only queries containing this text")
    page_contains: Optional[str] = Field(default=None, description="Only pages whose URL contains this text")
    country: Optional[str] = Field(default=None, description="Only this country, ISO 3166-1 alpha-3 lowercase, e.g. usa")
    device: Optional[str] = Field(default=None, description="Only this device: DESKTOP, MOBILE or TABLET")
    row_limit: int = Field(default=100, description="Maximum rows, 1-25000")
    start_row: int = Field(default=0, description="Row offset for paging")


class SearchAnalyticsRow(BaseAppOutput):
    keys: List[str] = Field(description="Values of the requested dimensions, in order")
    clicks: float = Field(description="Clicks")
    impressions: float = Field(description="Impressions")
    ctr: float = Field(description="Click-through rate, 0-1")
    position: float = Field(description="Average position")


class QuerySearchAnalyticsOutput(BaseAppOutput):
    rows: List[SearchAnalyticsRow] = Field(description="Result rows, sorted by clicks")


# --- inspect_url ---

class InspectUrlInput(BaseAppInput):
    site_url: str = Field(description="Property URL exactly as list_sites returns it")
    url: str = Field(description="Full URL of the page to inspect; must belong to the property")


class InspectUrlOutput(BaseAppOutput):
    verdict: str = Field(description="Overall index verdict: PASS, PARTIAL, FAIL or NEUTRAL")
    coverage_state: str = Field(description="Index coverage, e.g. 'Submitted and indexed'")
    indexing_state: str = Field(default="", description="Whether indexing is allowed")
    last_crawl_time: str = Field(default="", description="Last crawl time, RFC 3339")
    page_fetch_state: str = Field(default="", description="Result of the last fetch")
    google_canonical: str = Field(default="", description="Canonical URL Google chose")
    user_canonical: str = Field(default="", description="Canonical URL the page declares")
    mobile_usability_verdict: str = Field(default="", description="Mobile usability verdict")
    link: str = Field(default="", description="Link to the full report in Search Console")
    raw: Any = Field(description="Full inspection result from the API")


# --- list_sitemaps ---

class ListSitemapsInput(BaseAppInput):
    site_url: str = Field(description="Property URL exactly as list_sites returns it")


class Sitemap(BaseAppOutput):
    path: str = Field(description="Sitemap URL")
    last_submitted: str = Field(default="", description="When it was last submitted")
    last_downloaded: str = Field(default="", description="When Google last downloaded it")
    is_pending: bool = Field(default=False, description="Whether processing is pending")
    errors: int = Field(default=0, description="Number of errors")
    warnings: int = Field(default=0, description="Number of warnings")


class ListSitemapsOutput(BaseAppOutput):
    sitemaps: List[Sitemap] = Field(description="Sitemaps submitted for the property")


def _build_service():
    """Build the Search Console API service using the OAuth access token."""
    token = os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN")
    if not token:
        raise ValueError(
            "GOOGLE_OAUTH_ACCESS_TOKEN not found. "
            "Please ensure the Google Search Console integration is connected in Settings."
        )
    return build("searchconsole", "v1", credentials=Credentials(token=token))


class App(BaseApp):

    async def setup(self):
        self.logger = logging.getLogger(__name__)

    @function(credentials=CREDENTIALS, annotations={"title": "List sites", **READ_ONLY})
    async def list_sites(self, input_data: ListSitesInput) -> ListSitesOutput:
        """List the Search Console properties you can access."""
        self.logger.info("Listing Search Console sites")
        result = _build_service().sites().list().execute()
        sites = [
            Site(site_url=s["siteUrl"], permission_level=s.get("permissionLevel", ""))
            for s in result.get("siteEntry", [])
        ]
        self.logger.info(f"Found {len(sites)} sites")
        return ListSitesOutput(sites=sites)

    @function(credentials=CREDENTIALS, annotations={"title": "Query search analytics", **READ_ONLY})
    async def query_search_analytics(self, input_data: QuerySearchAnalyticsInput) -> QuerySearchAnalyticsOutput:
        """Clicks, impressions, CTR and position from Google Search, grouped by query, page, country, device or date."""
        filters = []
        for dimension, operator, value in (
            ("query", "contains", input_data.query_contains),
            ("page", "contains", input_data.page_contains),
            ("country", "equals", input_data.country),
            ("device", "equals", input_data.device),
        ):
            if value:
                filters.append({"dimension": dimension, "operator": operator, "expression": value})
        body = {
            "startDate": input_data.start_date,
            "endDate": input_data.end_date,
            "dimensions": input_data.dimensions,
            "type": input_data.search_type,
            "rowLimit": input_data.row_limit,
            "startRow": input_data.start_row,
        }
        if filters:
            body["dimensionFilterGroups"] = [{"filters": filters}]
        self.logger.info(f"Querying search analytics for {input_data.site_url} {input_data.start_date}..{input_data.end_date} by {input_data.dimensions}")
        result = _build_service().searchanalytics().query(siteUrl=input_data.site_url, body=body).execute()
        rows = [
            SearchAnalyticsRow(
                keys=r.get("keys", []),
                clicks=r.get("clicks", 0),
                impressions=r.get("impressions", 0),
                ctr=r.get("ctr", 0),
                position=r.get("position", 0),
            )
            for r in result.get("rows", [])
        ]
        self.logger.info(f"Got {len(rows)} rows")
        return QuerySearchAnalyticsOutput(rows=rows)

    @function(credentials=CREDENTIALS, annotations={"title": "Inspect URL", **READ_ONLY})
    async def inspect_url(self, input_data: InspectUrlInput) -> InspectUrlOutput:
        """Google's index status for one URL: coverage, last crawl, canonical, mobile usability."""
        self.logger.info(f"Inspecting {input_data.url}")
        result = _build_service().urlInspection().index().inspect(
            body={"inspectionUrl": input_data.url, "siteUrl": input_data.site_url}
        ).execute()
        r = result.get("inspectionResult", {})
        index = r.get("indexStatusResult", {})
        return InspectUrlOutput(
            verdict=index.get("verdict", ""),
            coverage_state=index.get("coverageState", ""),
            indexing_state=index.get("indexingState", ""),
            last_crawl_time=index.get("lastCrawlTime", ""),
            page_fetch_state=index.get("pageFetchState", ""),
            google_canonical=index.get("googleCanonical", ""),
            user_canonical=index.get("userCanonical", ""),
            mobile_usability_verdict=r.get("mobileUsabilityResult", {}).get("verdict", ""),
            link=r.get("inspectionResultLink", ""),
            raw=r,
        )

    @function(credentials=CREDENTIALS, annotations={"title": "List sitemaps", **READ_ONLY})
    async def list_sitemaps(self, input_data: ListSitemapsInput) -> ListSitemapsOutput:
        """List the sitemaps submitted for a property, with their status."""
        self.logger.info(f"Listing sitemaps for {input_data.site_url}")
        result = _build_service().sitemaps().list(siteUrl=input_data.site_url).execute()
        sitemaps = [
            Sitemap(
                path=s.get("path", ""),
                last_submitted=s.get("lastSubmitted", ""),
                last_downloaded=s.get("lastDownloaded", ""),
                is_pending=s.get("isPending", False),
                errors=int(s.get("errors", 0)),
                warnings=int(s.get("warnings", 0)),
            )
            for s in result.get("sitemap", [])
        ]
        self.logger.info(f"Found {len(sitemaps)} sitemaps")
        return ListSitemapsOutput(sitemaps=sitemaps)
