"""Outbound HTTP that only reaches public internet addresses.

Mirrors go/api/internal/lib/safehttp: the check runs at dial time, after DNS
resolution, on every connection the transport opens. A redirect opens a new
connection, so a public URL that redirects to an internal one is refused, and
a hostname that resolves to an internal address is refused however it is
spelled. The dial then goes to the checked IP itself, so the name is not
resolved a second time (DNS rebinding). TLS still verifies the certificate
against the hostname: httpcore passes the origin's host as server_hostname.

Proxies are never used: through a proxy the check would judge the proxy's
address instead of the destination's.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import Callable, Iterable, List, Optional, Sequence

import httpcore
import httpx


class BlockedAddressError(Exception):
    """The destination is not a public unicast address."""


# Ranges the ipaddress predicates do not already cover, or that we want named
# explicitly so the refusal says why. Same list as the Go package.
BLOCKED_NETWORKS = [
    ipaddress.ip_network(n)
    for n in (
        "0.0.0.0/8",  # "this network"; 0.x reaches the local host on Linux
        "100.64.0.0/10",  # carrier-grade NAT (Alibaba metadata is 100.100.100.200)
        "192.0.0.0/24",  # IETF protocol assignments
        "192.0.2.0/24",  # TEST-NET-1
        "198.18.0.0/15",  # benchmarking
        "198.51.100.0/24",  # TEST-NET-2
        "203.0.113.0/24",  # TEST-NET-3
        "240.0.0.0/4",  # reserved, includes broadcast
        "::/96",  # IPv4-compatible (deprecated): embeds an IPv4 address
        "64:ff9b::/96",  # NAT64: embeds an IPv4 address
        "64:ff9b:1::/48",  # local-use NAT64
        "2001::/32",  # Teredo: embeds an IPv4 address
        "2002::/16",  # 6to4: embeds an IPv4 address
        "2001:db8::/32",  # documentation
        "fc00::/7",  # unique local (fd00:ec2::254 is AWS metadata)
    )
]


# Every IPv6 address the internet routes today is in 2000::/3; the rest
# (fec0::/10 site-local, ::/8, ...) is reserved or local.
GLOBAL_UNICAST_V6 = ipaddress.ip_network("2000::/3")


def check_ip(ip: str) -> None:
    """Raise BlockedAddressError unless ip is a public unicast address.

    IPv4-mapped IPv6 (::ffff:a.b.c.d) is judged as the IPv4 it carries.
    """
    try:
        addr = ipaddress.ip_address(ip.split("%", 1)[0])
    except ValueError:
        raise BlockedAddressError(f"invalid address {ip!r}")
    if addr.version == 6 and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    if addr.is_loopback:
        raise BlockedAddressError(f"loopback {addr}")
    if addr.is_link_local:
        raise BlockedAddressError(f"link-local {addr}")
    if addr.is_unspecified:
        raise BlockedAddressError(f"unspecified {addr}")
    if addr.is_multicast:
        raise BlockedAddressError(f"multicast {addr}")
    for net in BLOCKED_NETWORKS:
        if addr.version == net.version and addr in net:
            raise BlockedAddressError(f"reserved {addr} ({net})")
    if addr.is_private:
        raise BlockedAddressError(f"private {addr}")
    if addr.version == 6 and addr not in GLOBAL_UNICAST_V6:
        raise BlockedAddressError(f"non-global {addr}")
    if addr.is_reserved or not addr.is_global:
        raise BlockedAddressError(f"non-public {addr}")


# A resolver takes (host, port) and returns IP strings. Tests replace it.
Resolver = Callable[[str, int], Sequence[str]]


def _system_resolve(host: str, port: int) -> List[str]:
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return [info[4][0] for info in infos]


resolver: Resolver = _system_resolve


def resolve_checked(host: str, port: int) -> List[str]:
    """Resolve host and return its addresses, IPv4 first, if every one is public.

    Every address must pass: a name with one internal answer is refused, since
    which answer a later connection gets is not ours to choose.
    """
    host = host.strip("[]")
    try:
        ips = list(dict.fromkeys(resolver(host, port)))
    except socket.gaierror as e:
        raise httpcore.ConnectError(f"could not resolve {host}: {e}") from e
    if not ips:
        raise httpcore.ConnectError(f"no addresses for {host}")
    for ip in ips:
        try:
            check_ip(ip)
        except BlockedAddressError as e:
            raise BlockedAddressError(f"blocked destination {host}: {e}") from None
    # Containers often have no IPv6 route; try IPv4 answers first.
    return sorted(ips, key=lambda ip: ":" in ip)


class GuardedAsyncBackend(httpcore.AsyncNetworkBackend):
    """Resolves and checks the host, then dials the checked IP."""

    def __init__(self, inner: Optional[httpcore.AsyncNetworkBackend] = None):
        self._inner = inner or httpcore.AnyIOBackend()

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        loop = asyncio.get_running_loop()
        ips = await loop.run_in_executor(None, resolve_checked, host, port)
        last: Optional[Exception] = None
        for ip in ips:
            try:
                return await self._inner.connect_tcp(
                    ip, port, timeout=timeout, local_address=local_address, socket_options=socket_options
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout, OSError) as e:
                last = e
        raise last  # type: ignore[misc]

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise BlockedAddressError("unix sockets are not allowed")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


class GuardedSyncBackend(httpcore.NetworkBackend):
    def __init__(self, inner: Optional[httpcore.NetworkBackend] = None):
        self._inner = inner or httpcore.SyncBackend()

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        last: Optional[Exception] = None
        for ip in resolve_checked(host, port):
            try:
                return self._inner.connect_tcp(
                    ip, port, timeout=timeout, local_address=local_address, socket_options=socket_options
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout, OSError) as e:
                last = e
        raise last  # type: ignore[misc]

    def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise BlockedAddressError("unix sockets are not allowed")

    def sleep(self, seconds: float) -> None:
        self._inner.sleep(seconds)


def _ssl_context():
    return httpx.create_ssl_context(verify=True, trust_env=False)


class GuardedAsyncTransport(httpx.AsyncHTTPTransport):
    """httpx transport whose every connection goes only to a public address."""

    def __init__(self, backend: Optional[httpcore.AsyncNetworkBackend] = None, **pool_kwargs):
        super().__init__(trust_env=False)
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=_ssl_context(),
            network_backend=backend or GuardedAsyncBackend(),
            max_connections=pool_kwargs.get("max_connections", 50),
            max_keepalive_connections=pool_kwargs.get("max_keepalive_connections", 20),
            keepalive_expiry=pool_kwargs.get("keepalive_expiry", 30.0),
        )


class GuardedSyncTransport(httpx.HTTPTransport):
    def __init__(self, backend: Optional[httpcore.NetworkBackend] = None):
        super().__init__(trust_env=False)
        self._pool = httpcore.ConnectionPool(
            ssl_context=_ssl_context(),
            network_backend=backend or GuardedSyncBackend(),
        )


def blocked_cause(exc: BaseException) -> Optional[BlockedAddressError]:
    """The BlockedAddressError behind an exception httpx may have wrapped."""
    seen = 0
    while exc is not None and seen < 10:
        if isinstance(exc, BlockedAddressError):
            return exc
        exc = exc.__cause__ or exc.__context__
        seen += 1
    return None


# Request headers that say how to answer or what the body is, not who is
# asking. On a redirect to another origin only these are kept.
PLAIN_HEADERS = {
    "content-type", "accept", "accept-charset", "accept-encoding", "accept-language",
    "user-agent", "cache-control", "pragma", "range", "if-modified-since", "if-none-match",
}

# Connection-level headers a caller must not set: the HTTP library owns them.
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-connection", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length", "expect",
}


def clean_request_headers(headers: Iterable[tuple]) -> dict:
    """Drop hop-by-hop headers (and any the Connection header names)."""
    items = [(str(k), str(v)) for k, v in headers]
    named = set()
    for k, v in items:
        if k.lower() == "connection":
            named.update(t.strip().lower() for t in v.split(",") if t.strip())
    out = {}
    for k, v in items:
        lk = k.lower()
        if lk in HOP_BY_HOP or lk in named:
            continue
        out[k] = v
    return out
