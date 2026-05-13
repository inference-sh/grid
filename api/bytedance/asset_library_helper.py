"""
BytePlus Asset Library helper for private virtual portrait library.
Uses AK/SK authentication via byteplussdkcore universal API.

Manages asset groups and assets for Seedance 2.0 Studio apps.
Assets uploaded here get asset:// URIs usable in video generation.
"""

import os
import asyncio
import hashlib
import logging
from typing import Optional, Dict, Any, List

from byteplussdkcore import Configuration, ApiClient
from byteplussdkcore.universal import UniversalApi, UniversalInfo


def setup_asset_client(
    ak: Optional[str] = None,
    sk: Optional[str] = None,
    region: str = "ap-southeast-1",
) -> UniversalApi:
    """
    Configure BytePlus universal API client with AK/SK credentials.

    Args:
        ak: Access key. Falls back to BYTEPLUS_AK env var.
        sk: Secret key. Falls back to BYTEPLUS_SK env var.
        region: BytePlus region.

    Returns:
        Configured UniversalApi instance.
    """
    access_key = ak or os.environ.get("BYTEPLUS_AK")
    secret_key = sk or os.environ.get("BYTEPLUS_SK")
    if not access_key or not secret_key:
        raise RuntimeError(
            "BYTEPLUS_AK and BYTEPLUS_SK environment variables are required "
            "for asset library access."
        )

    config = Configuration()
    config.ak = access_key
    config.sk = secret_key
    config.region = region
    config.scheme = "https"

    client = ApiClient(config)
    return UniversalApi(client)


def _ark_call(api: UniversalApi, action: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Make a call to the ARK service via the universal API."""
    info = UniversalInfo(
        method="POST",
        service="ark",
        version="2024-01-01",
        action=action,
        content_type="application/json",
    )
    return api.do_call(info, body)


def create_asset_group(
    api: UniversalApi,
    name: str,
    description: str = "",
    project_name: str = "default",
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Create an asset group in the private virtual portrait library.

    Args:
        api: Configured UniversalApi client.
        name: Name for the asset group.
        description: Description of the asset group.
        project_name: BytePlus project name.
        logger: Optional logger.

    Returns:
        Asset group ID (e.g. "group-20260318033332-xxxxx").
    """
    body = {
        "Name": name,
        "Description": description,
        "GroupType": "AIGC",
        "ProjectName": project_name,
    }
    if logger:
        logger.info(f"Creating asset group: {name}")

    resp = _ark_call(api, "CreateAssetGroup", body)
    group_id = resp.get("Id") or resp.get("id")
    if not group_id:
        raise RuntimeError(f"Failed to create asset group, response: {resp}")

    if logger:
        logger.info(f"Asset group created: {group_id}")
    return group_id


def create_asset(
    api: UniversalApi,
    group_id: str,
    url: str,
    asset_type: str = "Image",
    name: str = "",
    skip_moderation: bool = False,
    project_name: str = "default",
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Upload an asset to the private virtual portrait library.

    Args:
        api: Configured UniversalApi client.
        group_id: Target asset group ID.
        url: Accessible URL of the asset file.
        asset_type: Type of asset: "Image", "Video", or "Audio".
        name: Optional name for the asset.
        skip_moderation: Whether to skip content pre-filter review.
        project_name: BytePlus project name.
        logger: Optional logger.

    Returns:
        Asset ID (e.g. "asset-20260318071009-xxxxx").
    """
    body: Dict[str, Any] = {
        "GroupId": group_id,
        "URL": url,
        "AssetType": asset_type,
        "ProjectName": project_name,
    }
    if name:
        body["Name"] = name
    if skip_moderation:
        body["Moderation"] = {"Strategy": "Skip"}

    if logger:
        logger.info(f"Uploading {asset_type} asset to group {group_id}")

    resp = _ark_call(api, "CreateAsset", body)
    asset_id = resp.get("Id") or resp.get("id")
    if not asset_id:
        raise RuntimeError(f"Failed to create asset, response: {resp}")

    if logger:
        logger.info(f"Asset created: {asset_id}")
    return asset_id


def _url_to_asset_name(url: str) -> str:
    """Derive a stable asset name from a URL for deduplication."""
    return f"inf:{hashlib.sha256(url.encode()).hexdigest()[:16]}"


def find_active_asset(
    api: UniversalApi,
    group_id: str,
    name: str,
    project_name: str = "default",
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """
    Find an existing Active asset by name within a group.

    Returns:
        Asset ID if found and Active, None otherwise.
    """
    if logger:
        logger.info(f"Checking for existing asset with name: {name}")
    resp = _ark_call(api, "ListAssets", {
        "Filter": {
            "GroupIds": [group_id],
            "GroupType": "AIGC",
            "Name": name,
            "Statuses": ["Active"],
        },
        "PageNumber": 1,
        "PageSize": 1,
    })
    items = resp.get("Items") or resp.get("items") or []
    for item in items:
        item_name = item.get("Name") or item.get("name") or ""
        if item_name == name:
            asset_id = item.get("Id") or item.get("id")
            if logger:
                logger.info(f"Found existing active asset: {asset_id}")
            return asset_id
    if logger:
        logger.info("No existing asset found, will upload new")
    return None


def get_asset(
    api: UniversalApi,
    asset_id: str,
    project_name: str = "default",
) -> Dict[str, Any]:
    """
    Get asset information by ID.

    Returns:
        Asset dict with Status, URL, AssetType, etc.
    """
    return _ark_call(api, "GetAsset", {
        "Id": asset_id,
        "ProjectName": project_name,
    })


async def wait_for_asset_active(
    api: UniversalApi,
    asset_id: str,
    project_name: str = "default",
    logger: Optional[logging.Logger] = None,
    poll_interval: float = 2.0,
    timeout: float = 300.0,
) -> str:
    """
    Poll an asset until its status becomes Active.

    Args:
        api: Configured UniversalApi client.
        asset_id: Asset ID to poll.
        project_name: BytePlus project name.
        logger: Optional logger.
        poll_interval: Seconds between polls.
        timeout: Maximum wait time in seconds.

    Returns:
        The asset ID (confirmed active).

    Raises:
        RuntimeError: If asset fails processing or times out.
    """
    elapsed = 0.0
    while elapsed < timeout:
        info = get_asset(api, asset_id, project_name=project_name)
        status = info.get("Status") or info.get("status")

        if status == "Active":
            if logger:
                logger.info(f"Asset {asset_id} is active")
            return asset_id
        elif status == "Failed":
            raise RuntimeError(f"Asset {asset_id} processing failed")
        else:
            if logger and int(elapsed) % 10 == 0:
                logger.info(f"Asset {asset_id} status: {status}, waiting...")
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

    raise RuntimeError(f"Asset {asset_id} timed out after {timeout}s")


async def upload_and_activate(
    api: UniversalApi,
    group_id: str,
    url: str,
    asset_type: str = "Image",
    skip_moderation: bool = False,
    project_name: str = "default",
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Upload an asset and wait for it to become Active.
    Deduplicates by URL: derives a stable name from the URL hash and
    checks for an existing Active asset before uploading.

    Returns:
        Asset URI in the format "asset://asset-xxxxx".
    """
    asset_name = _url_to_asset_name(url)

    existing_id = find_active_asset(
        api, group_id, asset_name, project_name=project_name, logger=logger,
    )
    if existing_id:
        if logger:
            logger.info(f"REUSING existing asset {existing_id} for: {url[:80]}")
        return f"asset://{existing_id}"

    if logger:
        logger.info(f"UPLOADING new asset for: {url[:80]}")
    asset_id = create_asset(
        api, group_id, url,
        asset_type=asset_type,
        name=asset_name,
        skip_moderation=skip_moderation,
        project_name=project_name,
        logger=logger,
    )
    await wait_for_asset_active(
        api, asset_id,
        project_name=project_name,
        logger=logger,
    )
    return f"asset://{asset_id}"


async def upload_images_to_library(
    api: UniversalApi,
    group_id: str,
    image_urls: List[str],
    project_name: str = "default",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:
    """
    Upload multiple images to the asset library concurrently.

    Args:
        api: Configured UniversalApi client.
        group_id: Target asset group ID.
        image_urls: List of image URLs to upload.
        project_name: BytePlus project name.
        logger: Optional logger.

    Returns:
        Dict mapping original URL → asset URI (asset://asset-xxxxx).
    """
    if not image_urls:
        return {}

    if logger:
        logger.info(f"Uploading {len(image_urls)} images to asset library")

    tasks = []
    for url in image_urls:
        tasks.append(upload_and_activate(
            api, group_id, url,
            asset_type="Image",
            project_name=project_name,
            logger=logger,
        ))

    asset_uris = await asyncio.gather(*tasks)
    url_to_asset = dict(zip(image_urls, asset_uris))

    if logger:
        logger.info(f"All {len(image_urls)} assets are active")
    return url_to_asset
