"""
Fastly Domain Research API.

Search for available domains and check registration/aftermarket status.
"""

import os
import asyncio
import logging
import httpx
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, OutputMeta, RawMeta
from pydantic import Field, BaseModel
from typing import Optional, List


class DomainSuggestion(BaseModel):
    domain: str = Field(description="Full suggested domain (subdomain + zone)")
    subdomain: str = Field(description="Subdomain portion")
    zone: str = Field(description="Zone (TLD) portion")
    path: Optional[str] = Field(default=None, description="Path to append to complete the suggestion")


class Offer(BaseModel):
    currency: Optional[str] = Field(default=None, description="Currency for the offer")
    price: Optional[str] = Field(default=None, description="Price from the aftermarket vendor")
    vendor: Optional[str] = Field(default=None, description="Aftermarket vendor name")


class DomainStatus(BaseModel):
    domain: str = Field(description="The queried domain")
    status: str = Field(description="Space-delimited status values (rightmost = highest priority)")
    zone: str = Field(description="Zone of the domain")
    tags: Optional[str] = Field(default=None, description="Space-delimited tags")
    scope: Optional[str] = Field(default=None, description="Scope of the status check")
    offers: Optional[List[Offer]] = Field(default=None, description="Aftermarket offers if available")


class StatusInput(BaseAppInput):
    domains: List[str] = Field(description="Domains to check (e.g. ['maku.com', 'maku.sh'])")
    scope: Optional[str] = Field(default=None, description="Set to 'estimated' for DNS/aftermarket-only check (cheaper), omit for precise registry-level check")


class StatusOutput(BaseAppOutput):
    results: List[DomainStatus] = Field(description="Status for each queried domain")


class SuggestInput(BaseAppInput):
    query: str = Field(description="Search terms for domain suggestions")


class SuggestOutput(BaseAppOutput):
    results: List[DomainSuggestion] = Field(description="Suggested domains")


BASE_URL = "https://api.domainr.com/domain-management/v1/tools"


class App(BaseApp):
    async def setup(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = os.environ["FASTLY_KEY"]
        self.client = httpx.AsyncClient(timeout=30, headers={"Fastly-Key": self.api_key})
        self.logger.info("Fastly Domain Research initialized")

    async def suggest(self, input_data: SuggestInput) -> SuggestOutput:
        self.logger.info(f"Suggesting domains for: {input_data.query}")

        response = await self.client.get(
            f"{BASE_URL}/suggest",
            params={"query": input_data.query},
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", []):
            results.append(DomainSuggestion(
                domain=item.get("domain", ""),
                subdomain=item.get("subdomain", ""),
                zone=item.get("zone", ""),
                path=item.get("path"),
            ))

        self.logger.info(f"Got {len(results)} suggestions")

        return SuggestOutput(
            results=results,
            output_meta=OutputMeta(outputs=[RawMeta(cost=0.001)]),
        )

    async def status(self, input_data: StatusInput) -> StatusOutput:
        self.logger.info(f"Checking status for {len(input_data.domains)} domains")

        async def check_one(domain: str) -> DomainStatus:
            params = {"domain": domain}
            if input_data.scope:
                params["scope"] = input_data.scope

            response = await self.client.get(
                f"{BASE_URL}/status",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            offers = None
            if "offers" in data:
                offers = [
                    Offer(
                        currency=o.get("currency"),
                        price=o.get("price"),
                        vendor=o.get("vendor"),
                    )
                    for o in data["offers"]
                ]

            return DomainStatus(
                domain=data.get("domain", domain),
                status=data.get("status", "unknown"),
                zone=data.get("zone", ""),
                tags=data.get("tags"),
                scope=data.get("scope"),
                offers=offers,
            )

        results = await asyncio.gather(*[check_one(d) for d in input_data.domains])

        self.logger.info(f"Checked {len(results)} domains")

        cost_per = 0.05 if input_data.scope == "estimated" else 0.1

        return StatusOutput(
            results=results,
            output_meta=OutputMeta(outputs=[RawMeta(cost=cost_per * len(input_data.domains))]),
        )

    async def unload(self):
        await self.client.aclose()
