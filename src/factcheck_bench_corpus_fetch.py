"""Safe, resumable source-document fetching for Study I.

The corpus is *benchmark-grounded*: URLs come from the evaluation-only URL
manifest, but claim/evidence labels are never used by this module.  Fetching is
deliberately separated from passage construction so that the exact raw bytes
and clean document text can be audited and frozen before retrieval parameters
are selected.

This module does not use requests/urllib's automatic redirect handling.  Every
initial URL and redirect target is validated, DNS-resolved, checked for public
addresses, and then connected to using one of those checked addresses.  This
prevents a redirect or DNS rebinding from turning the corpus builder into an
SSRF client for local/private services.
"""

from __future__ import annotations

import email.utils
import hashlib
import http.client
import io
import ipaddress
import json
import logging
import platform
import re
import signal
import socket
import ssl
import threading
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser


LOGGER = logging.getLogger(__name__)

FETCH_SCHEMA_VERSION = "fcb_source_document_fetch_v2"
DOCUMENT_SCHEMA_VERSION = "fcb_source_document_v2"
EXTRACTOR_SUITE_VERSION = "fcb_safe_document_extractors_v3"
HTML_EXTRACTOR_VERSION = "stdlib_readable_html_jsonld_v3"
PLAIN_TEXT_EXTRACTOR_VERSION = "stdlib_plain_text_v1"
JSON_EXTRACTOR_VERSION = "stdlib_json_scalar_v1"
PDF_EXTRACTOR_VERSION = "pypdf_text_v1"
PYTHON_RUNTIME = "%s-%s" % (platform.python_implementation(), platform.python_version())
try:
    PYPDF_RUNTIME_VERSION = importlib_metadata.version("pypdf")
except importlib_metadata.PackageNotFoundError:
    PYPDF_RUNTIME_VERSION = "not-installed"

REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
RETRYABLE_CODES = frozenset({429, 500, 502, 503, 504})
SUPPORTED_SCHEMES = frozenset({"http", "https"})

_WHITESPACE_RE = re.compile(r"[\t\x0b\x0c\r ]+")
_BLANK_LINES_RE = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")
_SAFE_DOC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_BOILERPLATE_RE = re.compile(
    r"(?:^|[-_\s])(?:nav(?:igation)?|footer|header|menu|cookie|consent|"
    r"advert(?:isement|ising)?|social|share|breadcrumb|promo|subscribe|"
    r"modal|popup|sidebar|toolbar|newsletter)(?:$|[-_\s])",
    re.IGNORECASE,
)
_BLOCK_TAGS = frozenset(
    {"p", "li", "blockquote", "pre", "dd", "dt", "figcaption", "caption"}
)
_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_ALWAYS_IGNORED_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "svg",
        "canvas",
        "template",
        "nav",
        "footer",
        "aside",
        "form",
        "button",
        "input",
        "select",
        "option",
    }
)
_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)
_BOILERPLATE_CONTAINER_TAGS = frozenset(
    {"div", "section", "aside", "nav", "header", "footer", "form"}
)
_MIN_HTML_CLEAN_CHARS = 80
_MIN_HTML_WORDS = 10


class FetchFailure(Exception):
    """A classified failure safe to persist in an audit record."""

    def __init__(
        self,
        category: str,
        message: str,
        *,
        status_code: Optional[int] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.message = message
        self.status_code = status_code
        self.details = dict(details or {})


class URLWallClockTimeout(TimeoutError):
    """Raised when all work for one manifest URL exceeds its total budget."""


@dataclass(frozen=True)
class FetchSettings:
    user_agent: str
    robots_user_agent: str
    timeout_seconds: float
    max_retries: int
    backoff_seconds: float
    per_host_delay_seconds: float
    max_redirects: int
    max_response_bytes: int
    checkpoint_every: int
    max_retry_wait_seconds: float
    url_wall_clock_timeout_seconds: float


@dataclass
class HTTPResult:
    requested_url: str
    final_url: str
    status_code: int
    reason: str
    headers: Dict[str, str]
    body: bytes
    redirect_chain: List[Dict[str, Any]]
    resolved_addresses: List[Dict[str, Any]]
    attempt_count: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _format_progress_bytes(value: Any) -> str:
    """Return a compact human-readable byte count for terminal progress."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return "unknown size"
    if value < 1024:
        return f"{value} B"
    if value < 1024 * 1024:
        return f"{value / 1024:.0f} KB"
    return f"{value / (1024 * 1024):.1f} MB"


def _progress_target(url: Optional[str]) -> str:
    """Return a short, non-sensitive display target for a manifest URL."""
    if not url:
        return "<missing URL>"
    try:
        parsed = urlsplit(url)
    except ValueError:
        return "<invalid URL>"
    return parsed.hostname or "<invalid URL>"


def _print_fetch_progress(position: int, total: int, message: str) -> None:
    """Emit unbuffered progress so long-running fetches remain observable."""
    print(f"[{position}/{total}] {message}", flush=True)


def _fetch_result_message(document: Mapping[str, Any], *, reused: bool = False) -> str:
    status = str(document.get("fetch_status") or "unknown")
    size = _format_progress_bytes(document.get("raw_byte_count"))
    elapsed_attempts = document.get("attempt_count")
    attempt_suffix = (
        f", {elapsed_attempts} HTTP attempt(s)"
        if isinstance(elapsed_attempts, int) and elapsed_attempts > 1
        else ""
    )
    if status == "success":
        reuse_suffix = ", reused frozen content" if reused else ""
        return f"success, {size}{reuse_suffix}{attempt_suffix}"
    status_code = document.get("status_code")
    code_suffix = f" (HTTP {status_code})" if isinstance(status_code, int) else ""
    error = document.get("error")
    message = error.get("message") if isinstance(error, Mapping) else None
    if isinstance(message, str) and message.strip():
        compact = " ".join(message.split())
        if len(compact) > 180:
            compact = compact[:177] + "..."
        return f"{status}{code_suffix}: {compact}{attempt_suffix}"
    return f"{status}{code_suffix}{attempt_suffix}"


@contextmanager
def _url_wall_clock_timeout(seconds: float):
    """Bound one URL's robots, redirects, retries, and response work on Unix.

    Fetching is currently a synchronous CLI operation on the main thread.  On
    platforms without ``setitimer`` (or if called from a worker thread), the
    existing socket-level timeouts remain the fallback.
    """
    if (
        seconds <= 0
        or not hasattr(signal, "SIGALRM")
        or not hasattr(signal, "setitimer")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def raise_timeout(signum: int, frame: Any) -> None:
        del signum, frame
        raise URLWallClockTimeout(
            f"Total URL processing time exceeded {seconds:g} seconds"
        )

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            remaining = max(0.001, previous_timer[0] - (time.monotonic() - started))
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _normalise_space(text: str) -> str:
    lines = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = _WHITESPACE_RE.sub(" ", unescape(raw_line)).strip()
        if line:
            lines.append(line)
        elif lines and lines[-1] != "":
            lines.append("")
    normalised = "\n".join(lines).strip()
    return _BLANK_LINES_RE.sub("\n\n", normalised)


def _config_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    section = config.get("fetch")
    if isinstance(section, Mapping):
        return section
    return config


def _settings_from_config(config: Mapping[str, Any]) -> FetchSettings:
    section = _config_section(config)

    def integer(name: str, default: int, minimum: int) -> int:
        value = section.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"fetch.{name} must be an integer >= {minimum}")
        return value

    def number(name: str, default: float, minimum: float) -> float:
        value = section.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"fetch.{name} must be numeric")
        result = float(value)
        if result < minimum:
            raise ValueError(f"fetch.{name} must be >= {minimum}")
        return result

    user_agent = section.get(
        "user_agent", "FactCheckBenchCorpusBuilder/1.0 (+research; no-auth)"
    )
    robots_user_agent = section.get(
        "robots_user_agent", "FactCheckBenchCorpusBuilder"
    )
    if not isinstance(user_agent, str) or not user_agent.strip():
        raise ValueError("fetch.user_agent must be a non-empty string")
    if not isinstance(robots_user_agent, str) or not robots_user_agent.strip():
        raise ValueError("fetch.robots_user_agent must be a non-empty string")
    if "\r" in user_agent or "\n" in user_agent:
        raise ValueError("fetch.user_agent must not contain newlines")

    return FetchSettings(
        user_agent=user_agent.strip(),
        robots_user_agent=robots_user_agent.strip(),
        timeout_seconds=number("timeout_seconds", 20.0, 0.1),
        max_retries=integer("max_retries", 2, 0),
        backoff_seconds=number("backoff_seconds", 1.0, 0.0),
        per_host_delay_seconds=number("per_host_delay_seconds", 0.5, 0.0),
        max_redirects=integer("max_redirects", 8, 0),
        max_response_bytes=integer("max_response_bytes", 25 * 1024 * 1024, 1),
        checkpoint_every=integer("checkpoint_every", 25, 1),
        max_retry_wait_seconds=number("max_retry_wait_seconds", 30.0, 0.0),
        # This default is intentionally derived rather than written into the
        # frozen config mid-run: adding it to the current config would change
        # the resume fingerprint and invalidate already fetched documents.
        url_wall_clock_timeout_seconds=number(
            "url_wall_clock_timeout_seconds", 90.0, 1.0
        ),
    )


def _manifest_identity(row: Mapping[str, Any], index: int) -> str:
    for key in ("url_id", "url_hash", "canonical_url_hash"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    candidate = row.get("canonical_url") or row.get("raw_url")
    if isinstance(candidate, str) and candidate.strip():
        return "url_" + hashlib.sha256(candidate.strip().encode("utf-8")).hexdigest()[:20]
    return "url_row_%08d" % (index + 1)


def _doc_id(row: Mapping[str, Any], manifest_id: str) -> str:
    existing = row.get("doc_id")
    if isinstance(existing, str) and existing.strip():
        candidate = existing.strip()
        if not _SAFE_DOC_ID_RE.fullmatch(candidate):
            raise ValueError("Unsafe or invalid doc_id in URL manifest: %r" % candidate)
        return candidate
    return "doc_" + hashlib.sha256(manifest_id.encode("utf-8")).hexdigest()[:20]


def _url_from_manifest(row: Mapping[str, Any]) -> Optional[str]:
    canonical_status = row.get("canonicalisation_status")
    if canonical_status is not None and canonical_status != "ok":
        return None
    canonical = row.get("canonical_url")
    raw = row.get("raw_url")
    if isinstance(canonical, str) and canonical.strip():
        return canonical.strip()
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _basic_normalise_url(url: str) -> str:
    """Normalise only syntax required for safe fetching.

    Stage A3/A4 owns research canonicalisation.  This helper does not remove
    query parameters and must not be used to create qrels or ranking inputs.
    """
    stripped = url.strip()
    if any(ord(character) < 32 or ord(character) == 127 for character in stripped):
        raise FetchFailure("unsafe_url", "URL contains ASCII control characters")
    try:
        split = urlsplit(stripped)
        port = split.port
    except (TypeError, ValueError) as exc:
        raise FetchFailure("unsafe_url", "Malformed URL: %s" % exc)
    scheme = split.scheme.lower()
    if scheme not in SUPPORTED_SCHEMES:
        raise FetchFailure("unsafe_url", "Only http/https URLs are allowed")
    if split.username is not None or split.password is not None:
        raise FetchFailure("unsafe_url", "URLs containing userinfo are refused")
    if not split.hostname:
        raise FetchFailure("unsafe_url", "URL has no hostname")
    try:
        hostname = split.hostname.rstrip(".").encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise FetchFailure("unsafe_url", "Invalid international hostname: %s" % exc)
    if not hostname or hostname == "localhost" or hostname.endswith(".localhost"):
        raise FetchFailure("unsafe_url", "Localhost targets are refused")
    if "%" in hostname:
        raise FetchFailure("unsafe_url", "Scoped/zone-qualified addresses are refused")
    default_port = 443 if scheme == "https" else 80
    effective_port = port or default_port
    if not (1 <= effective_port <= 65535):
        raise FetchFailure("unsafe_url", "URL port is outside 1..65535")
    display_host = "[%s]" % hostname if ":" in hostname else hostname
    netloc = display_host
    if port is not None and port != default_port:
        netloc += ":%d" % port
    path = split.path or "/"
    return urlunsplit((scheme, netloc, path, split.query, ""))


def _is_public_address(address: str) -> bool:
    try:
        value = ipaddress.ip_address(address)
    except ValueError:
        return False
    # Some Python/ipaddress versions report multicast space as ``is_global``;
    # reject each unsafe class explicitly as well as requiring global routing.
    unsafe = (
        value.is_loopback
        or value.is_private
        or value.is_link_local
        or value.is_multicast
        or value.is_reserved
        or value.is_unspecified
    )
    if isinstance(value, ipaddress.IPv6Address) and value.ipv4_mapped is not None:
        mapped = value.ipv4_mapped
        unsafe = unsafe or not _is_public_address(str(mapped))
    return bool(value.is_global and not unsafe)


def _resolve_public(url: str) -> Tuple[str, int, List[str]]:
    normalised = _basic_normalise_url(url)
    split = urlsplit(normalised)
    assert split.hostname is not None
    host = split.hostname
    port = split.port or (443 if split.scheme == "https" else 80)

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        addresses = [literal.compressed]
    else:
        try:
            answers = socket.getaddrinfo(
                host,
                port,
                family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise FetchFailure("dns_error", "DNS resolution failed: %s" % exc)
        addresses = []
        for answer in answers:
            address = answer[4][0]
            if address not in addresses:
                addresses.append(address)
    if not addresses:
        raise FetchFailure("dns_error", "DNS returned no usable addresses")
    unsafe = [address for address in addresses if not _is_public_address(address)]
    if unsafe:
        raise FetchFailure(
            "unsafe_url",
            "DNS returned non-public address(es); target refused",
            details={"blocked_addresses": unsafe},
        )
    return host, port, addresses


def _target_path(url: str) -> str:
    split = urlsplit(url)
    target = split.path or "/"
    if split.query:
        target += "?" + split.query
    return target


def _header_map(headers: Sequence[Tuple[str, str]]) -> Dict[str, str]:
    grouped: MutableMapping[str, List[str]] = defaultdict(list)
    for name, value in headers:
        grouped[name.lower()].append(value.strip())
    return {name: ", ".join(values) for name, values in grouped.items()}


def _safe_response_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    """Persist useful provenance headers, never cookies/auth challenges."""
    allowed = {
        "accept-ranges",
        "cache-control",
        "content-disposition",
        "content-encoding",
        "content-language",
        "content-length",
        "content-type",
        "date",
        "etag",
        "expires",
        "last-modified",
        "location",
        "vary",
    }
    return {key: value for key, value in headers.items() if key in allowed}


def _read_limited(response: http.client.HTTPResponse, limit: int) -> bytes:
    content_length = response.getheader("Content-Length")
    if content_length:
        try:
            declared = int(content_length)
        except ValueError:
            declared = None
        if declared is not None and declared > limit:
            raise FetchFailure(
                "response_too_large",
                "Declared response size exceeds configured maximum",
                status_code=response.status,
                details={"content_length": declared, "max_response_bytes": limit},
            )
    chunks = []
    total = 0
    while True:
        chunk = response.read(min(64 * 1024, limit + 1 - total))
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise FetchFailure(
                "response_too_large",
                "Downloaded response exceeds configured maximum",
                status_code=response.status,
                details={"max_response_bytes": limit},
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _open_checked_connection(
    url: str,
    settings: FetchSettings,
    *,
    max_bytes: Optional[int] = None,
) -> Tuple[int, str, Dict[str, str], bytes, List[str]]:
    """Make one non-redirecting GET using a DNS-checked, pinned IP address."""
    normalised = _basic_normalise_url(url)
    split = urlsplit(normalised)
    host, port, addresses = _resolve_public(normalised)
    errors = []
    for address in addresses:
        connection: Optional[http.client.HTTPConnection] = None
        raw_socket: Optional[socket.socket] = None
        try:
            raw_socket = socket.create_connection(
                (address, port), timeout=settings.timeout_seconds
            )
            raw_socket.settimeout(settings.timeout_seconds)
            if split.scheme == "https":
                context = ssl.create_default_context()
                wrapped = context.wrap_socket(raw_socket, server_hostname=host)
                raw_socket = None
                connection = http.client.HTTPSConnection(
                    host, port=port, timeout=settings.timeout_seconds, context=context
                )
                connection.sock = wrapped
            else:
                connection = http.client.HTTPConnection(
                    host, port=port, timeout=settings.timeout_seconds
                )
                connection.sock = raw_socket
                raw_socket = None
            connection.request(
                "GET",
                _target_path(normalised),
                headers={
                    "User-Agent": settings.user_agent,
                    "Accept": (
                        "text/html,application/pdf,text/plain,"
                        "application/json;q=0.9,*/*;q=0.1"
                    ),
                    "Accept-Encoding": "identity",
                    "Connection": "close",
                },
            )
            response = connection.getresponse()
            headers = _header_map(response.getheaders())
            body = _read_limited(response, max_bytes or settings.max_response_bytes)
            return response.status, response.reason or "", headers, body, addresses
        except FetchFailure:
            raise
        except (socket.timeout, TimeoutError) as exc:
            errors.append("%s: timeout (%s)" % (address, exc))
        except ssl.SSLError as exc:
            errors.append("%s: TLS error (%s)" % (address, exc))
        except (OSError, http.client.HTTPException) as exc:
            errors.append("%s: network error (%s)" % (address, exc))
        finally:
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass
            if raw_socket is not None:
                try:
                    raw_socket.close()
                except OSError:
                    pass
    message = "; ".join(errors) or "No public address could be connected"
    if errors and all("timeout" in item for item in errors):
        raise FetchFailure("timeout", message)
    if errors and all("TLS error" in item for item in errors):
        raise FetchFailure("tls_error", message)
    raise FetchFailure("network_error", message)


def _retry_delay(
    headers: Mapping[str, str], settings: FetchSettings, retry: int
) -> Optional[float]:
    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            seconds = float(retry_after)
        except ValueError:
            try:
                when = email.utils.parsedate_to_datetime(retry_after)
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                seconds = max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                seconds = -1.0
        if seconds >= 0:
            return (
                seconds
                if seconds <= settings.max_retry_wait_seconds
                else None
            )
    calculated = settings.backoff_seconds * (2 ** retry)
    return (
        calculated
        if calculated <= settings.max_retry_wait_seconds
        else None
    )


def _origin_key(url: str) -> str:
    split = urlsplit(_basic_normalise_url(url))
    default = 443 if split.scheme == "https" else 80
    port = split.port or default
    assert split.hostname is not None
    return "%s://%s:%d" % (split.scheme, split.hostname, port)


def _throttle(
    origin: str,
    settings: FetchSettings,
    last_request_by_origin: MutableMapping[str, float],
    minimum_delay_by_origin: Optional[Mapping[str, float]] = None,
) -> None:
    declared_delay = (
        float(minimum_delay_by_origin.get(origin, 0.0))
        if minimum_delay_by_origin is not None
        else 0.0
    )
    effective_delay = max(settings.per_host_delay_seconds, declared_delay)
    previous = last_request_by_origin.get(origin)
    if previous is not None and effective_delay > 0:
        remaining = effective_delay - (time.monotonic() - previous)
        if remaining > 0:
            time.sleep(remaining)
    last_request_by_origin[origin] = time.monotonic()


def _fetch_http(
    url: str,
    settings: FetchSettings,
    last_request_by_origin: MutableMapping[str, float],
    *,
    max_bytes: Optional[int] = None,
    redirect_authorizer: Optional[Callable[[str], Optional[Mapping[str, Any]]]] = None,
    minimum_delay_by_origin: Optional[Mapping[str, float]] = None,
) -> HTTPResult:
    requested = _basic_normalise_url(url)
    current = requested
    redirects: List[Dict[str, Any]] = []
    resolved_audit: List[Dict[str, Any]] = []
    total_attempts = 0

    def preserve_progress(exc: FetchFailure, current_url: str) -> FetchFailure:
        """Attach the full redirect/request progress before propagating failure."""
        exc.details.setdefault("redirect_chain", [dict(row) for row in redirects])
        exc.details.setdefault("current_url", current_url)
        exc.details.setdefault("attempt_count", total_attempts)
        exc.details.setdefault(
            "resolved_addresses", [dict(row) for row in resolved_audit]
        )
        return exc

    for redirect_number in range(settings.max_redirects + 1):
        origin = _origin_key(current)
        final_status = 0
        final_reason = ""
        final_headers: Dict[str, str] = {}
        final_body = b""
        for retry in range(settings.max_retries + 1):
            _throttle(
                origin,
                settings,
                last_request_by_origin,
                minimum_delay_by_origin,
            )
            total_attempts += 1
            try:
                status, reason, headers, body, addresses = _open_checked_connection(
                    current, settings, max_bytes=max_bytes
                )
            except FetchFailure as exc:
                if (
                    exc.category in {"timeout", "network_error", "tls_error"}
                    and retry < settings.max_retries
                ):
                    delay = settings.backoff_seconds * (2 ** retry)
                    if delay <= settings.max_retry_wait_seconds:
                        time.sleep(delay)
                        continue
                raise preserve_progress(exc, current)
            resolved_audit.append(
                {
                    "url": current,
                    "redirect_index": redirect_number,
                    "addresses": addresses,
                }
            )
            final_status, final_reason = status, reason
            final_headers, final_body = headers, body
            if status in RETRYABLE_CODES and retry < settings.max_retries:
                delay = _retry_delay(headers, settings, retry)
                if delay is not None:
                    time.sleep(delay)
                    continue
            break

        if final_status in REDIRECT_CODES:
            location = final_headers.get("location")
            if not location:
                raise FetchFailure(
                    "redirect_error",
                    "Redirect response has no Location header",
                    status_code=final_status,
                    details={
                        "redirect_chain": list(redirects),
                        "current_url": current,
                        "attempt_count": total_attempts,
                        "resolved_addresses": list(resolved_audit),
                    },
                )
            if redirect_number >= settings.max_redirects:
                raise FetchFailure(
                    "redirect_error",
                    "Maximum redirect count exceeded",
                    status_code=final_status,
                    details={
                        "redirect_chain": list(redirects),
                        "current_url": current,
                        "attempt_count": total_attempts,
                        "resolved_addresses": list(resolved_audit),
                    },
                )
            next_url = _basic_normalise_url(urljoin(current, location))
            redirect_record: Dict[str, Any] = {
                "status_code": final_status,
                "from_url": current,
                "to_url": next_url,
            }
            if redirect_authorizer is not None:
                try:
                    authorization = redirect_authorizer(next_url)
                except FetchFailure as exc:
                    redirect_record["authorization"] = exc.details.get("robots")
                    redirects.append(redirect_record)
                    raise preserve_progress(exc, next_url)
                if authorization is not None:
                    redirect_record["authorization"] = dict(authorization)
            # Validation and DNS resolution happen again when the next request
            # is opened.  Syntax/userinfo/local literal checks happen now too.
            redirects.append(redirect_record)
            current = next_url
            continue

        return HTTPResult(
            requested_url=requested,
            final_url=current,
            status_code=final_status,
            reason=final_reason,
            headers=final_headers,
            body=final_body,
            redirect_chain=redirects,
            resolved_addresses=resolved_audit,
            attempt_count=total_attempts,
        )

    raise FetchFailure(
        "redirect_error",
        "Unreachable redirect state",
        details={
            "redirect_chain": list(redirects),
            "current_url": current,
            "attempt_count": total_attempts,
            "resolved_addresses": list(resolved_audit),
        },
    )


def _robots_url(url: str) -> str:
    split = urlsplit(_basic_normalise_url(url))
    return urlunsplit((split.scheme, split.netloc, "/robots.txt", "", ""))


def _robots_allowed(
    url: str,
    settings: FetchSettings,
    robots_cache: MutableMapping[str, Dict[str, Any]],
    robots_delay_by_origin: MutableMapping[str, float],
    last_request_by_origin: MutableMapping[str, float],
) -> Tuple[bool, Dict[str, Any]]:
    origin = _origin_key(url)
    cached = robots_cache.get(origin)
    if cached is not None:
        allowed = bool(cached["allowed_for_url"](url))
        audit = dict(cached["audit"])
        audit.update({"requested_url": url, "allowed": allowed, "cache_hit": True})
        return allowed, audit

    robots_target = _robots_url(url)
    audit: Dict[str, Any] = {"robots_url": robots_target, "checked_at": _utc_now()}
    try:
        result = _fetch_http(
            robots_target,
            settings,
            last_request_by_origin,
            max_bytes=min(settings.max_response_bytes, 512 * 1024),
            minimum_delay_by_origin=robots_delay_by_origin,
        )
    except FetchFailure as exc:
        # A network/DNS/timeout failure is not an instruction from a site, but
        # conservatively avoid fetching when no cached policy is available.
        audit.update({"status": "unavailable", "error_category": exc.category})
        allowed_function = lambda candidate: False
    else:
        audit.update(
            {
                "status_code": result.status_code,
                "final_url": result.final_url,
                "redirect_chain": result.redirect_chain,
            }
        )
        if result.status_code in {404, 410}:
            audit["status"] = "not_present_allow"
            allowed_function = lambda candidate: True
        elif result.status_code in {401, 403, 429} or result.status_code >= 500:
            audit["status"] = "unavailable_disallow"
            allowed_function = lambda candidate: False
        elif 200 <= result.status_code < 300:
            parser = RobotFileParser()
            parser.set_url(result.final_url)
            text = result.body.decode("utf-8", errors="replace")
            parser.parse(text.splitlines())
            crawl_delay = parser.crawl_delay(settings.robots_user_agent)
            request_rate = parser.request_rate(settings.robots_user_agent)
            request_rate_interval = None
            if (
                request_rate is not None
                and request_rate.requests > 0
                and request_rate.seconds >= 0
            ):
                request_rate_interval = (
                    float(request_rate.seconds) / float(request_rate.requests)
                )
            effective_delay = max(
                float(crawl_delay or 0.0),
                float(request_rate_interval or 0.0),
            )
            audit["robots_pacing"] = {
                "crawl_delay_seconds": crawl_delay,
                "request_rate": (
                    {
                        "requests": request_rate.requests,
                        "seconds": request_rate.seconds,
                    }
                    if request_rate is not None
                    else None
                ),
                "request_rate_minimum_interval_seconds": request_rate_interval,
                "effective_minimum_delay_seconds": effective_delay,
            }
            audit["status"] = "parsed"
            allowed_function = lambda candidate, parser=parser: parser.can_fetch(
                settings.robots_user_agent, candidate
            )
        else:
            audit["status"] = "unexpected_status_disallow"
            allowed_function = lambda candidate: False

    robots_cache[origin] = {
        "allowed_for_url": allowed_function,
        "audit": audit,
    }
    allowed = bool(allowed_function(url))
    returned_audit = dict(audit)
    returned_audit.update(
        {"requested_url": url, "allowed": allowed, "cache_hit": False}
    )
    return allowed, returned_audit


def _register_robots_delay(
    url: str,
    audit: Mapping[str, Any],
    minimum_delay_by_origin: MutableMapping[str, float],
) -> None:
    pacing = audit.get("robots_pacing")
    if not isinstance(pacing, Mapping):
        return
    delay = pacing.get("effective_minimum_delay_seconds")
    if isinstance(delay, bool) or not isinstance(delay, (int, float)) or delay <= 0:
        return
    origin = _origin_key(url)
    minimum_delay_by_origin[origin] = max(
        float(delay), float(minimum_delay_by_origin.get(origin, 0.0))
    )


def _require_robots_authorization(url: str, allowed: bool, audit: Mapping[str, Any]) -> None:
    if allowed:
        return
    robots_error = audit.get("error_category")
    category = (
        robots_error
        if robots_error
        in {"dns_error", "timeout", "network_error", "tls_error", "unsafe_url"}
        else "robots_disallowed"
    )
    raise FetchFailure(
        str(category),
        "robots.txt policy could not authorize this URL",
        details={"robots": dict(audit)},
    )


class _ReadableHTMLParser(HTMLParser):
    """Extract ordered readable blocks without executing or rendering HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.blocks: List[Dict[str, Any]] = []
        self.current_section: Optional[str] = None
        self._ignored_depth = 0
        self._element_stack: List[Tuple[str, bool]] = []
        self._title_depth = 0
        self._title_parts: List[str] = []
        self._capture_tag: Optional[str] = None
        self._capture_parts: List[str] = []
        self._capture_kind: Optional[str] = None
        self._table_depth = 0
        self._row_depth = 0
        self._cell_depth = 0
        self._cell_parts: List[str] = []
        self._row_cells: List[str] = []
        self._bare_parts: List[str] = []
        self._bare_kind: Optional[str] = None

    @staticmethod
    def _is_boilerplate(tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> bool:
        if tag in _ALWAYS_IGNORED_TAGS:
            return True
        # Never discard a whole document because a root/semantic container has
        # a class such as "navigation-enabled" or "header-present".  This was
        # the cause of complete extraction failure for frozen Wikipedia HTML.
        if tag not in _BOILERPLATE_CONTAINER_TAGS:
            return False
        attributes = {key.lower(): (value or "") for key, value in attrs}
        marker = " ".join(
            [attributes.get("id", ""), attributes.get("class", ""), attributes.get("role", "")]
        )
        return bool(marker and _BOILERPLATE_RE.search(marker))

    def _fallback_kind(self) -> Optional[str]:
        active_tags = [tag for tag, ignored in self._element_stack if not ignored]
        if any(tag in {"main", "article"} for tag in active_tags):
            return "main_bare_text"
        if "div" in active_tags:
            return "div_fallback_text"
        return None

    def _flush_bare(self) -> None:
        text = _normalise_space("".join(self._bare_parts))
        if text:
            self.blocks.append(
                {
                    "kind": self._bare_kind or "bare_text",
                    "section": self.current_section,
                    "text": text,
                }
            )
        self._bare_parts = []
        self._bare_kind = None

    def finish(self) -> None:
        """Flush readable text left by malformed/unclosed container markup."""
        self._flush_bare()

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        ignored_here = self._ignored_depth > 0 or self._is_boilerplate(tag, attrs)
        if self._ignored_depth == 0 and (
            ignored_here
            or tag in _HEADING_TAGS
            or tag in _BLOCK_TAGS
            or tag in {"main", "article", "div", "table"}
        ):
            self._flush_bare()
        if tag not in _VOID_TAGS:
            self._element_stack.append((tag, ignored_here))
        if ignored_here:
            if tag not in _VOID_TAGS:
                self._ignored_depth += 1
            return
        if tag == "title":
            self._title_depth += 1
            return
        if tag == "table":
            self._table_depth += 1
            return
        if self._table_depth:
            if tag == "tr":
                self._row_depth += 1
                self._row_cells = []
            elif tag in {"td", "th"} and self._row_depth:
                self._cell_depth += 1
                self._cell_parts = []
            return
        if self._capture_tag is None and (tag in _HEADING_TAGS or tag in _BLOCK_TAGS):
            self._capture_tag = tag
            self._capture_parts = []
            self._capture_kind = "heading" if tag in _HEADING_TAGS else "paragraph"
        elif tag == "br" and self._capture_tag is not None:
            self._capture_parts.append("\n")
        elif tag == "br" and self._fallback_kind() is not None:
            self._bare_parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._title_depth:
            self._title_parts.append(data)
            return
        if self._table_depth and self._cell_depth:
            self._cell_parts.append(data)
        elif self._capture_tag is not None:
            self._capture_parts.append(data)
        else:
            fallback_kind = self._fallback_kind()
            if fallback_kind is not None:
                if self._bare_kind is not None and self._bare_kind != fallback_kind:
                    self._flush_bare()
                self._bare_kind = fallback_kind
                self._bare_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._ignored_depth == 0 and tag in {"main", "article", "div"}:
            self._flush_bare()
        ignored_closed = False
        matching_index = None
        for index in range(len(self._element_stack) - 1, -1, -1):
            if self._element_stack[index][0] == tag:
                matching_index = index
                break
        if matching_index is not None:
            closing = self._element_stack[matching_index:]
            del self._element_stack[matching_index:]
            ignored_count = sum(ignored for _, ignored in closing)
            if ignored_count:
                self._ignored_depth = max(0, self._ignored_depth - ignored_count)
                ignored_closed = True
        if ignored_closed:
            return
        if tag == "title" and self._title_depth:
            self._title_depth -= 1
            if not self._title_depth:
                self.title = _normalise_space("".join(self._title_parts))
            return
        if self._table_depth:
            if tag in {"td", "th"} and self._cell_depth:
                cell = _normalise_space("".join(self._cell_parts))
                if cell:
                    self._row_cells.append(cell)
                self._cell_depth = max(0, self._cell_depth - 1)
                self._cell_parts = []
            elif tag == "tr" and self._row_depth:
                row = " | ".join(self._row_cells)
                if row:
                    self.blocks.append(
                        {"kind": "table_row", "section": self.current_section, "text": row}
                    )
                self._row_depth = max(0, self._row_depth - 1)
                self._row_cells = []
            elif tag == "table":
                self._table_depth = max(0, self._table_depth - 1)
            return
        if tag == self._capture_tag:
            text = _normalise_space("".join(self._capture_parts))
            if text:
                if self._capture_kind == "heading":
                    self.current_section = text
                self.blocks.append(
                    {"kind": self._capture_kind, "section": self.current_section, "text": text}
                )
            self._capture_tag = None
            self._capture_kind = None
            self._capture_parts = []


class _JSONLDArticleParser(HTMLParser):
    """Collect JSON-LD script payloads without executing page JavaScript."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.payloads: List[str] = []
        self._capturing = False
        self._parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag.casefold() != "script":
            return
        attributes = {key.casefold(): (value or "") for key, value in attrs}
        content_type = attributes.get("type", "").casefold().split(";", 1)[0].strip()
        if content_type == "application/ld+json":
            self._capturing = True
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._capturing:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "script" and self._capturing:
            payload = "".join(self._parts).strip()
            if payload:
                self.payloads.append(payload)
            self._capturing = False
            self._parts = []


def _walk_jsonld_articles(value: Any) -> List[Tuple[Optional[str], str]]:
    candidates: List[Tuple[Optional[str], str]] = []
    if isinstance(value, Mapping):
        headline = value.get("headline") or value.get("name")
        title = _normalise_space(str(headline)) if headline else None
        body = value.get("articleBody")
        if isinstance(body, str) and body.strip():
            candidates.append((title, _normalise_space(body)))
        for child in value.values():
            if isinstance(child, (Mapping, list)):
                candidates.extend(_walk_jsonld_articles(child))
    elif isinstance(value, list):
        for child in value:
            candidates.extend(_walk_jsonld_articles(child))
    return candidates


def _jsonld_article_fallback(text: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    parser = _JSONLDArticleParser()
    try:
        parser.feed(text)
        parser.close()
    except Exception:
        return None, []
    candidates: List[Tuple[Optional[str], str]] = []
    for payload in parser.payloads:
        try:
            value = json.loads(payload)
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        candidates.extend(_walk_jsonld_articles(value))
    if not candidates:
        return None, []
    title, body = max(candidates, key=lambda item: len(item[1]))
    blocks: List[Dict[str, Any]] = []
    if title:
        blocks.append({"kind": "jsonld_headline", "section": title, "text": title})
    for paragraph in re.split(r"\n\s*\n", body):
        paragraph = _normalise_space(paragraph)
        if paragraph:
            blocks.append(
                {"kind": "jsonld_article_body", "section": title, "text": paragraph}
            )
    return title, blocks


def _charset_from_content_type(content_type: str) -> Optional[str]:
    match = re.search(r"(?:^|;)\s*charset\s*=\s*[\"']?([^;\"'\s]+)", content_type, re.I)
    return match.group(1).strip() if match else None


def _decode_text(payload: bytes, content_type: str) -> Tuple[str, str]:
    candidates = []
    declared = _charset_from_content_type(content_type)
    if declared:
        candidates.append(declared)
    candidates.extend(["utf-8", "utf-8-sig", "windows-1252", "latin-1"])
    tried = set()
    for encoding in candidates:
        lowered = encoding.lower()
        if lowered in tried:
            continue
        tried.add(lowered)
        try:
            return payload.decode(encoding), encoding
        except (LookupError, UnicodeDecodeError):
            continue
    return payload.decode("utf-8", errors="replace"), "utf-8-replacement"


def _json_blocks(value: Any, path: str = "$") -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            blocks.extend(_json_blocks(child, "%s.%s" % (path, key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            blocks.extend(_json_blocks(child, "%s[%d]" % (path, index)))
    elif value is not None:
        text = _normalise_space(str(value))
        if text:
            blocks.append(
                {"kind": "json_value", "section": path, "text": "%s: %s" % (path, text)}
            )
    return blocks


def _finalize_blocks(blocks: Sequence[Mapping[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Create clean text and exact offsets from ordered extracted blocks."""
    output: List[Dict[str, Any]] = []
    text_parts: List[str] = []
    cursor = 0
    for original in blocks:
        text = _normalise_space(str(original.get("text", "")))
        if not text:
            continue
        if text_parts:
            text_parts.append("\n\n")
            cursor += 2
        char_start = cursor
        text_parts.append(text)
        cursor += len(text)
        block = dict(original)
        block["text"] = text
        block["char_start"] = char_start
        block["char_end"] = cursor
        output.append(block)
    return "".join(text_parts), output


def _parse_html(payload: bytes, content_type: str) -> Dict[str, Any]:
    text, encoding = _decode_text(payload, content_type)
    parser = _ReadableHTMLParser()
    try:
        parser.feed(text)
        parser.close()
        parser.finish()
    except Exception as exc:  # HTMLParser can surface malformed entity errors.
        raise FetchFailure("parse_failure", "HTML parsing failed: %s" % exc)
    clean_text, blocks = _finalize_blocks(parser.blocks)
    jsonld_title, jsonld_blocks = _jsonld_article_fallback(text)
    jsonld_text, finalized_jsonld_blocks = _finalize_blocks(jsonld_blocks)
    use_jsonld = bool(jsonld_text) and (
        len(clean_text) < _MIN_HTML_CLEAN_CHARS
        or len(jsonld_text) > len(clean_text) * 1.5
    )
    if use_jsonld:
        clean_text = jsonld_text
        blocks = finalized_jsonld_blocks
    word_count = len(re.findall(r"\w+", clean_text, flags=re.UNICODE))
    # Keep concise but potentially decisive evidence pages.  Reject only when
    # both independent signals say that the extracted body is trivial.
    if len(clean_text) < _MIN_HTML_CLEAN_CHARS and word_count < _MIN_HTML_WORDS:
        raise FetchFailure(
            "empty_content",
            "HTML contained insufficient readable main text",
            details={
                "clean_char_count": len(clean_text),
                "word_count": word_count,
                "minimum_clean_chars": _MIN_HTML_CLEAN_CHARS,
                "minimum_words": _MIN_HTML_WORDS,
                "jsonld_article_found": bool(jsonld_text),
            },
        )
    return {
        "document_type": "html",
        "title": (jsonld_title if use_jsonld else parser.title) or None,
        "clean_text": clean_text,
        "content_blocks": blocks,
        "text_encoding": encoding,
        "page_count": None,
        "extractor_method": "stdlib_htmlparser_readable_blocks_jsonld_fallback",
        "extractor_version": HTML_EXTRACTOR_VERSION,
        "html_extraction_source": "jsonld_article" if use_jsonld else "readable_blocks",
    }


def _parse_plain(payload: bytes, content_type: str) -> Dict[str, Any]:
    text, encoding = _decode_text(payload, content_type)
    normalized = _normalise_space(text)
    if not normalized:
        raise FetchFailure("empty_content", "Plain-text response was empty")
    blocks = [
        {"kind": "paragraph", "section": None, "text": paragraph}
        for paragraph in re.split(r"\n\s*\n", normalized)
        if paragraph.strip()
    ]
    clean_text, blocks = _finalize_blocks(blocks)
    return {
        "document_type": "text",
        "title": None,
        "clean_text": clean_text,
        "content_blocks": blocks,
        "text_encoding": encoding,
        "page_count": None,
        "extractor_method": "stdlib_plain_text_blocks",
        "extractor_version": PLAIN_TEXT_EXTRACTOR_VERSION,
    }


def _parse_json(payload: bytes, content_type: str) -> Dict[str, Any]:
    text, encoding = _decode_text(payload, content_type)
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise FetchFailure("parse_failure", "JSON parsing failed: %s" % exc)
    blocks = _json_blocks(value)
    clean_text, blocks = _finalize_blocks(blocks)
    if not clean_text:
        raise FetchFailure("empty_content", "JSON response had no scalar text")
    return {
        "document_type": "json",
        "title": None,
        "clean_text": clean_text,
        "content_blocks": blocks,
        "text_encoding": encoding,
        "page_count": None,
        "extractor_method": "stdlib_json_scalar_walk",
        "extractor_version": JSON_EXTRACTOR_VERSION,
    }


def _parse_pdf(payload: bytes) -> Dict[str, Any]:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        raise FetchFailure(
            "parse_failure",
            "PDF fetched but optional dependency 'pypdf' is not installed",
            details={"missing_dependency": "pypdf"},
        )
    try:
        reader = PdfReader(io.BytesIO(payload), strict=False)
        if getattr(reader, "is_encrypted", False):
            try:
                result = reader.decrypt("")
            except Exception:
                result = 0
            if not result:
                raise FetchFailure("parse_failure", "Encrypted PDF cannot be read")
        blocks = []
        for page_number, page in enumerate(reader.pages, start=1):
            extracted = page.extract_text() or ""
            text = _normalise_space(extracted)
            if not text:
                continue
            for paragraph in re.split(r"\n\s*\n", text):
                if paragraph.strip():
                    blocks.append(
                        {
                            "kind": "pdf_text",
                            "section": None,
                            "page": page_number,
                            "text": paragraph.strip(),
                        }
                    )
        clean_text, blocks = _finalize_blocks(blocks)
        if not clean_text:
            raise FetchFailure(
                "empty_content", "PDF had no extractable text (OCR is not attempted)"
            )
        metadata = getattr(reader, "metadata", None)
        title = None
        if metadata is not None:
            try:
                candidate = metadata.get("/Title")
            except (AttributeError, KeyError, TypeError):
                candidate = None
            if isinstance(candidate, str) and candidate.strip():
                title = _normalise_space(candidate)
        return {
            "document_type": "pdf",
            "title": title,
            "clean_text": clean_text,
            "content_blocks": blocks,
            "text_encoding": None,
            "page_count": len(reader.pages),
            "extractor_method": "pypdf_page_text",
            "extractor_version": PDF_EXTRACTOR_VERSION,
        }
    except FetchFailure:
        raise
    except Exception as exc:
        raise FetchFailure("parse_failure", "PDF parsing failed: %s" % exc)


def _media_type(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


def _parse_document(payload: bytes, content_type: str) -> Dict[str, Any]:
    media_type = _media_type(content_type)
    prefix = payload[:512].lstrip().lower()
    if media_type == "application/pdf" or payload.startswith(b"%PDF-"):
        return _parse_pdf(payload)
    if media_type in {"text/html", "application/xhtml+xml"} or prefix.startswith(
        (b"<!doctype html", b"<html")
    ):
        return _parse_html(payload, content_type)
    if media_type in {"application/json", "text/json"} or media_type.endswith("+json"):
        return _parse_json(payload, content_type)
    if media_type.startswith("text/") or media_type in {
        "application/text",
        "application/x-ndjson",
        "application/ndjson",
    }:
        if media_type in {"application/x-ndjson", "application/ndjson"}:
            return _parse_plain(payload, content_type)
        return _parse_plain(payload, content_type)
    if media_type in {"", "application/octet-stream"}:
        if prefix.startswith((b"{", b"[")):
            return _parse_json(payload, content_type)
        if b"\x00" not in payload[:4096]:
            return _parse_plain(payload, content_type)
    raise FetchFailure(
        "unsupported_type",
        "Unsupported response Content-Type: %s" % (content_type or "missing"),
        details={"content_type": content_type or None},
    )


def _extension(content_type: str, payload: bytes) -> str:
    media = _media_type(content_type)
    if media == "application/pdf" or payload.startswith(b"%PDF-"):
        return ".pdf"
    if media in {"text/html", "application/xhtml+xml"}:
        return ".html"
    if media in {"application/json", "text/json"} or media.endswith("+json"):
        return ".json"
    if media.startswith("text/"):
        return ".txt"
    return ".bin"


def _project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve_raw_path(raw_path: Any, project_root: Path) -> Optional[Path]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def _fetch_pipeline_sha256(fetch_config_sha256: str) -> str:
    return _canonical_json_sha256(
        {
            "fetch_schema_version": FETCH_SCHEMA_VERSION,
            "document_schema_version": DOCUMENT_SCHEMA_VERSION,
            "extractor_suite_version": EXTRACTOR_SUITE_VERSION,
            "extractor_registry": _extractor_registry(),
            "fetch_config_sha256": fetch_config_sha256,
        }
    )


def _extractor_registry() -> Dict[str, Dict[str, str]]:
    return {
        "html": {
            "method": "stdlib_htmlparser_readable_blocks_jsonld_fallback",
            "version": HTML_EXTRACTOR_VERSION,
            "runtime": PYTHON_RUNTIME,
        },
        "text": {
            "method": "stdlib_plain_text_blocks",
            "version": PLAIN_TEXT_EXTRACTOR_VERSION,
            "runtime": PYTHON_RUNTIME,
        },
        "json": {
            "method": "stdlib_json_scalar_walk",
            "version": JSON_EXTRACTOR_VERSION,
            "runtime": PYTHON_RUNTIME,
        },
        "pdf": {
            "method": "pypdf_page_text",
            "version": PDF_EXTRACTOR_VERSION,
            "runtime": PYTHON_RUNTIME,
            "pypdf_distribution_version": PYPDF_RUNTIME_VERSION,
        },
    }


def _extractor_fingerprint(
    document_type: str, extractor_method: str, extractor_version: str
) -> str:
    return _canonical_json_sha256(
        {
            "extractor_suite_version": EXTRACTOR_SUITE_VERSION,
            "document_type": document_type,
            "extractor_method": extractor_method,
            "extractor_version": extractor_version,
            "extractor_runtime": _extractor_registry().get(document_type),
        }
    )


def _manifest_request_snapshot(
    row: Mapping[str, Any], requested_url: Optional[str]
) -> Dict[str, Any]:
    canonical = row.get("canonical_url")
    return {
        "requested_url": requested_url,
        "canonical_url": canonical if isinstance(canonical, str) else None,
        "canonicalisation_status": row.get("canonicalisation_status"),
        "canonicalisation_version": row.get("canonicalisation_version"),
    }


def _resume_metadata_matches(
    record: Mapping[str, Any],
    *,
    manifest_row: Mapping[str, Any],
    requested_url: Optional[str],
    fetch_config_sha256: str,
    fetch_pipeline_sha256: str,
) -> bool:
    request_snapshot = _manifest_request_snapshot(manifest_row, requested_url)
    expected_metadata = {
        "schema_version": DOCUMENT_SCHEMA_VERSION,
        "requested_url": requested_url,
        "canonical_url": request_snapshot["canonical_url"],
        "manifest_request_sha256": _canonical_json_sha256(request_snapshot),
        "fetch_config_sha256": fetch_config_sha256,
        "fetch_pipeline_sha256": fetch_pipeline_sha256,
        "extractor_suite_version": EXTRACTOR_SUITE_VERSION,
    }
    return not any(
        record.get(key) != value for key, value in expected_metadata.items()
    )


def _verified_resume_record(
    record: Mapping[str, Any],
    project_root: Path,
    *,
    manifest_row: Mapping[str, Any],
    requested_url: Optional[str],
    fetch_config_sha256: str,
    fetch_pipeline_sha256: str,
) -> bool:
    if record.get("fetch_status") != "success":
        return False
    if not _resume_metadata_matches(
        record,
        manifest_row=manifest_row,
        requested_url=requested_url,
        fetch_config_sha256=fetch_config_sha256,
        fetch_pipeline_sha256=fetch_pipeline_sha256,
    ):
        return False
    document_type = record.get("document_type")
    extractor = _extractor_registry().get(str(document_type))
    if extractor is None:
        return False
    extractor_method = record.get("extractor_method")
    extractor_version = record.get("extractor_version")
    if (
        extractor_method != extractor["method"]
        or extractor_version != extractor["version"]
        or record.get("extractor_environment") != extractor
        or record.get("extractor_fingerprint_sha256")
        != _extractor_fingerprint(
            str(document_type), str(extractor_method), str(extractor_version)
        )
    ):
        return False
    expected = record.get("raw_sha256")
    path = _resolve_raw_path(record.get("raw_path"), project_root)
    if not isinstance(expected, str) or not path or not path.is_file():
        return False
    try:
        path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return False
    try:
        actual = _sha256_bytes(path.read_bytes())
    except OSError:
        return False
    if actual != expected:
        return False
    clean_text = record.get("clean_text")
    clean_hash = record.get("clean_text_sha256")
    if not isinstance(clean_hash, str) or not isinstance(clean_text, str):
        return False
    if hashlib.sha256(clean_text.encode("utf-8")).hexdigest() != clean_hash:
        return False
    if record.get("content_hash") != clean_hash:
        return False
    return True


def _verified_resume_failure_record(
    record: Mapping[str, Any],
    project_root: Path,
    *,
    manifest_row: Mapping[str, Any],
    requested_url: Optional[str],
    fetch_config_sha256: str,
    fetch_pipeline_sha256: str,
) -> bool:
    """Validate a completed failure so resume does not retry it implicitly."""
    status = record.get("fetch_status")
    if not isinstance(status, str) or status in {
        "success",
        "pending",
        "stale_pending",
        "internal_error",
        "orphaned_success",
    }:
        return False
    if not _resume_metadata_matches(
        record,
        manifest_row=manifest_row,
        requested_url=requested_url,
        fetch_config_sha256=fetch_config_sha256,
        fetch_pipeline_sha256=fetch_pipeline_sha256,
    ):
        return False
    if not isinstance(record.get("fetched_at"), str):
        return False
    error = record.get("error")
    if not isinstance(error, Mapping) or error.get("category") != status:
        return False

    raw_path_value = record.get("raw_path")
    raw_sha256 = record.get("raw_sha256")
    if raw_path_value is None and raw_sha256 is None:
        return True
    if not isinstance(raw_sha256, str):
        return False
    path = _resolve_raw_path(raw_path_value, project_root)
    if path is None or not path.is_file():
        return False
    try:
        path.resolve().relative_to(project_root.resolve())
        return _sha256_bytes(path.read_bytes()) == raw_sha256
    except (OSError, ValueError):
        return False


def _evaluation_only_trace(row: Mapping[str, Any]) -> Dict[str, Any]:
    nested = row.get("trace")
    result: Dict[str, Any] = {}
    keys = (
        "evidence_ids",
        "matched_evidence_ids",
        "claim_ids",
        "response_ids",
        "splits",
        "evidence_id",
        "claim_id",
        "response_id",
        "split",
        "domain",
        "url_hash",
    )
    if isinstance(nested, Mapping):
        result.update({key: nested.get(key) for key in keys if key in nested})
    result.update({key: row.get(key) for key in keys if key in row})
    return {
        "field_usage": "evaluation_only_gold_url_mapping",
        "prohibited_uses": [
            "query_construction",
            "ranking",
            "retrieved_verifier_prompt",
            "passage_payload",
        ],
        "trace": result,
    }


def _failure_status(status_code: int) -> str:
    if status_code == 404:
        return "http_404"
    if status_code == 403:
        return "http_403"
    if status_code == 429:
        return "http_429"
    if status_code in {401, 407}:
        return "authentication_required"
    if status_code in {402, 451}:
        return "access_restricted"
    if 500 <= status_code <= 599:
        return "server_error"
    return "http_error"


def _base_document(
    row: Mapping[str, Any],
    manifest_id: str,
    doc_id: str,
    url: Optional[str],
    fetch_config_sha256: str,
    fetch_pipeline_sha256: str,
) -> Dict[str, Any]:
    request_snapshot = _manifest_request_snapshot(row, url)
    return {
        "schema_version": DOCUMENT_SCHEMA_VERSION,
        "doc_id": doc_id,
        "source_url_manifest_id": manifest_id,
        "requested_url": url,
        "canonical_url": row.get("canonical_url"),
        "raw_url": row.get("raw_url"),
        "evaluation_only_source_trace": _evaluation_only_trace(row),
        "manifest_request_sha256": _canonical_json_sha256(request_snapshot),
        "fetch_config_sha256": fetch_config_sha256,
        "fetch_pipeline_sha256": fetch_pipeline_sha256,
        "extractor_suite_version": EXTRACTOR_SUITE_VERSION,
        "fetched_at": None,
        "fetch_status": "pending",
        "status_code": None,
        "http_reason": None,
        "content_type": None,
        "content_encoding": None,
        "response_headers": {},
        "final_url": None,
        "redirect_chain": [],
        "resolved_addresses": [],
        "attempt_count": 0,
        "fetch_attempted": False,
        "raw_path": None,
        "raw_sha256": None,
        "raw_content_sha256": None,
        "content_hash": None,
        "raw_byte_count": None,
        "document_type": None,
        "title": None,
        "clean_text": None,
        "clean_text_sha256": None,
        "clean_char_count": 0,
        "content_blocks": [],
        "text_encoding": None,
        "page_count": None,
        "extractor_method": None,
        "extractor_version": None,
        "extractor_environment": None,
        "extractor_fingerprint_sha256": None,
        "robots": None,
        "error": None,
        "failure_current_url": None,
        "fetch_reused_from_doc_id": None,
        "canonical_url_duplicate_group": None,
        "final_url_duplicate_group": None,
        "content_duplicate_group": None,
        "is_content_duplicate": False,
        "duplicate_component_id": None,
        "duplicate_component_primary_doc_id": None,
        "duplicate_relation_matches": [],
        "duplicate_of_doc_id": None,
        "is_duplicate": False,
    }


def _clone_success(source: Mapping[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    copied = dict(base)
    for key in (
        "fetched_at",
        "fetch_status",
        "status_code",
        "http_reason",
        "content_type",
        "content_encoding",
        "response_headers",
        "final_url",
        "redirect_chain",
        "resolved_addresses",
        "attempt_count",
        "fetch_attempted",
        "raw_path",
        "raw_sha256",
        "raw_content_sha256",
        "content_hash",
        "raw_byte_count",
        "document_type",
        "title",
        "clean_text",
        "clean_text_sha256",
        "clean_char_count",
        "content_blocks",
        "text_encoding",
        "page_count",
        "extractor_method",
        "extractor_version",
        "extractor_environment",
        "extractor_fingerprint_sha256",
        "robots",
        "error",
        "failure_current_url",
    ):
        copied[key] = source.get(key)
    copied["fetch_attempted"] = False
    copied["fetch_reused_from_doc_id"] = source.get("doc_id")
    return copied


def _fetch_one(
    row: Mapping[str, Any],
    manifest_id: str,
    doc_id: str,
    url: Optional[str],
    project_root: Path,
    raw_documents_dir: Path,
    settings: FetchSettings,
    robots_cache: MutableMapping[str, Dict[str, Any]],
    robots_delay_by_origin: MutableMapping[str, float],
    last_request_by_origin: MutableMapping[str, float],
    atomic_write_bytes: Any,
    fetch_config_sha256: str,
    fetch_pipeline_sha256: str,
) -> Dict[str, Any]:
    document = _base_document(
        row,
        manifest_id,
        doc_id,
        url,
        fetch_config_sha256,
        fetch_pipeline_sha256,
    )
    fetched_at = _utc_now()
    document["fetched_at"] = fetched_at
    document["fetch_attempted"] = True
    if url is None:
        if row.get("canonicalisation_status") not in (None, "ok"):
            category = "canonicalisation_invalid"
            message = "Manifest URL did not pass deterministic canonicalisation"
        else:
            category = "missing_url"
            message = "Manifest row has no URL"
        document["fetch_status"] = category
        document["error"] = {
            "category": category,
            "message": message,
            "details": {"canonicalisation_error": row.get("canonicalisation_error")},
        }
        return document

    try:
        normalised = _basic_normalise_url(url)
        allowed, robots_audit = _robots_allowed(
            normalised,
            settings,
            robots_cache,
            robots_delay_by_origin,
            last_request_by_origin,
        )
        robots_hops: List[Dict[str, Any]] = [
            {"url": normalised, "hop": "initial", "audit": robots_audit}
        ]
        document["robots"] = {"hops": robots_hops}
        _require_robots_authorization(normalised, allowed, robots_audit)
        _register_robots_delay(
            normalised, robots_audit, robots_delay_by_origin
        )

        def authorize_redirect(target: str) -> Mapping[str, Any]:
            redirect_allowed, redirect_audit = _robots_allowed(
                target,
                settings,
                robots_cache,
                robots_delay_by_origin,
                last_request_by_origin,
            )
            robots_hops.append(
                {"url": target, "hop": "redirect", "audit": redirect_audit}
            )
            _require_robots_authorization(target, redirect_allowed, redirect_audit)
            _register_robots_delay(
                target, redirect_audit, robots_delay_by_origin
            )
            return {"robots": redirect_audit, "allowed": True}

        result = _fetch_http(
            normalised,
            settings,
            last_request_by_origin,
            redirect_authorizer=authorize_redirect,
            minimum_delay_by_origin=robots_delay_by_origin,
        )
        document.update(
            {
                "status_code": result.status_code,
                "http_reason": result.reason,
                "content_type": result.headers.get("content-type"),
                "content_encoding": result.headers.get("content-encoding"),
                "response_headers": _safe_response_headers(result.headers),
                "final_url": result.final_url,
                "redirect_chain": result.redirect_chain,
                "resolved_addresses": result.resolved_addresses,
                "attempt_count": result.attempt_count,
            }
        )

        content_type = result.headers.get("content-type", "")
        if result.body:
            raw_sha = _sha256_bytes(result.body)
            raw_path = raw_documents_dir / (
                "%s_%s%s" % (doc_id, raw_sha[:16], _extension(content_type, result.body))
            )
            atomic_write_bytes(raw_path, result.body)
            document.update(
                {
                    "raw_path": _project_relative(raw_path, project_root),
                    "raw_sha256": raw_sha,
                    "raw_content_sha256": raw_sha,
                    "raw_byte_count": len(result.body),
                }
            )

        if not (200 <= result.status_code < 300):
            status = _failure_status(result.status_code)
            raise FetchFailure(
                status,
                "HTTP %d %s" % (result.status_code, result.reason),
                status_code=result.status_code,
            )
        if not result.body:
            raise FetchFailure(
                "empty_content",
                "HTTP response body was empty",
                status_code=result.status_code,
            )

        content_encodings = [
            value.strip().casefold()
            for value in result.headers.get("content-encoding", "").split(",")
            if value.strip()
        ]
        unsupported_encodings = [
            value for value in content_encodings if value != "identity"
        ]
        if unsupported_encodings:
            raise FetchFailure(
                "unsupported_content_encoding",
                "Server returned a non-identity Content-Encoding despite the request",
                status_code=result.status_code,
                details={"content_encodings": content_encodings},
            )

        parsed = _parse_document(result.body, content_type)
        clean_text = parsed["clean_text"]
        clean_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
        extractor_fingerprint = _extractor_fingerprint(
            str(parsed["document_type"]),
            str(parsed["extractor_method"]),
            str(parsed["extractor_version"]),
        )
        document.update(parsed)
        document.update(
            {
                "fetch_status": "success",
                "clean_text_sha256": clean_hash,
                "content_hash": clean_hash,
                "clean_char_count": len(clean_text),
                "extractor_environment": _extractor_registry()[
                    str(parsed["document_type"])
                ],
                "extractor_fingerprint_sha256": extractor_fingerprint,
                "error": None,
            }
        )
    except FetchFailure as exc:
        if not document.get("redirect_chain") and isinstance(
            exc.details.get("redirect_chain"), list
        ):
            document["redirect_chain"] = exc.details["redirect_chain"]
        if isinstance(exc.details.get("resolved_addresses"), list):
            document["resolved_addresses"] = exc.details["resolved_addresses"]
        if isinstance(exc.details.get("attempt_count"), int):
            document["attempt_count"] = exc.details["attempt_count"]
        if isinstance(exc.details.get("current_url"), str):
            document["failure_current_url"] = exc.details["current_url"]
        document["fetch_status"] = exc.category
        if document.get("status_code") is None:
            document["status_code"] = exc.status_code
        document["error"] = {
            "category": exc.category,
            "message": exc.message,
            "details": exc.details,
        }
    except URLWallClockTimeout:
        # The outer per-manifest-row controller records this as a completed,
        # resumable failure and proceeds to the next URL.
        raise
    except Exception as exc:  # Last-resort audit record; do not hide the row.
        LOGGER.exception("Unexpected corpus fetch failure for %s", url)
        document["fetch_status"] = "internal_error"
        document["error"] = {
            "category": "internal_error",
            "message": "%s: %s" % (type(exc).__name__, exc),
            "details": {},
        }
    return document


def _assign_duplicate_groups(documents: List[Dict[str, Any]]) -> None:
    """Assign flat duplicate components across URL and content relations.

    A relation in any of the three namespaces joins the same connected
    component.  Every alias points directly to one active, non-orphan,
    successful document.  If a component has no such document it is retained
    for audit, but no unsafe primary/alias is invented.
    """
    fields = (
        ("canonical_url", "canonical_url_duplicate_group", "canonical"),
        ("final_url", "final_url_duplicate_group", "final"),
        ("content_hash", "content_duplicate_group", "content"),
    )
    doc_ids = [str(document.get("doc_id") or "") for document in documents]
    if any(not doc_id for doc_id in doc_ids) or len(doc_ids) != len(set(doc_ids)):
        raise ValueError("Document records require unique, non-empty doc_id values")
    by_doc_id = dict(zip(doc_ids, documents))
    parent = {doc_id: doc_id for doc_id in doc_ids}

    def find(doc_id: str) -> str:
        root = doc_id
        while parent[root] != root:
            root = parent[root]
        while parent[doc_id] != doc_id:
            next_doc = parent[doc_id]
            parent[doc_id] = root
            doc_id = next_doc
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        # Root choice is deterministic, independent of row order.
        lower, upper = sorted((left_root, right_root))
        parent[upper] = lower

    for document in documents:
        document["canonical_url_duplicate_group"] = None
        document["final_url_duplicate_group"] = None
        document["content_duplicate_group"] = None
        document["is_content_duplicate"] = False
        document["duplicate_component_id"] = None
        document["duplicate_component_primary_doc_id"] = None
        document["duplicate_relation_matches"] = []
        document["duplicate_of_doc_id"] = None
        document["is_duplicate"] = False
    for value_field, group_field, prefix in fields:
        groups: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
        for document in documents:
            value = document.get(value_field)
            if isinstance(value, str) and value:
                groups[value].append(document)
        for value, members in groups.items():
            if len(members) < 2:
                for member in members:
                    member[group_field] = None
                continue
            ordered = sorted(members, key=lambda row: str(row["doc_id"]))
            group_id = "%s_%s" % (
                prefix, hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
            )
            first_doc_id = str(ordered[0]["doc_id"])
            for other in ordered[1:]:
                union(first_doc_id, str(other["doc_id"]))
            for member in ordered:
                member[group_field] = group_id
                if group_field == "content_duplicate_group":
                    member["is_content_duplicate"] = True
                member["duplicate_relation_matches"].append(
                    {
                        "relation": prefix,
                        "group_id": group_id,
                        "value_sha256": hashlib.sha256(
                            value.encode("utf-8")
                        ).hexdigest(),
                        "member_count": len(ordered),
                    }
                )

    components: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    for doc_id, document in by_doc_id.items():
        components[find(doc_id)].append(document)
    for members in components.values():
        if len(members) < 2:
            continue
        member_ids = sorted(str(member["doc_id"]) for member in members)
        component_id = "duplicate_component_" + hashlib.sha256(
            "\n".join(member_ids).encode("utf-8")
        ).hexdigest()[:16]
        eligible_primaries = [
            member
            for member in members
            if member.get("fetch_status") == "success"
            and not member.get("orphaned_from_url_manifest")
        ]
        eligible_primaries.sort(
            key=lambda member: (
                member.get("fetch_reused_from_doc_id") is not None,
                str(member["doc_id"]),
            )
        )
        primary_doc_id = (
            str(eligible_primaries[0]["doc_id"])
            if eligible_primaries
            else None
        )
        for member in members:
            member["is_duplicate"] = True
            member["duplicate_component_id"] = component_id
            member["duplicate_component_primary_doc_id"] = primary_doc_id
            if primary_doc_id is not None and member["doc_id"] != primary_doc_id:
                member["duplicate_of_doc_id"] = primary_doc_id

    # Defensive invariants: aliases are flat and can only target a usable
    # primary.  These checks make a regression fail before artifacts are saved.
    for document in documents:
        alias = document.get("duplicate_of_doc_id")
        if alias is None:
            continue
        target = by_doc_id.get(str(alias))
        if (
            target is None
            or target.get("fetch_status") != "success"
            or target.get("orphaned_from_url_manifest")
            or target.get("duplicate_of_doc_id") is not None
        ):
            raise AssertionError("Duplicate alias does not target a flat active success")


def _manifest_fetch_update(row: Mapping[str, Any], document: Mapping[str, Any]) -> Dict[str, Any]:
    updated = dict(row)
    for key in (
        "doc_id",
        "fetch_status",
        "fetched_at",
        "status_code",
        "content_type",
        "content_encoding",
        "final_url",
        "redirect_chain",
        "raw_path",
        "raw_sha256",
        "raw_content_sha256",
        "content_hash",
        "clean_text_sha256",
        "canonical_url_duplicate_group",
        "final_url_duplicate_group",
        "content_duplicate_group",
        "duplicate_component_id",
        "duplicate_component_primary_doc_id",
        "duplicate_relation_matches",
        "duplicate_of_doc_id",
        "is_duplicate",
        "is_content_duplicate",
        "fetch_reused_from_doc_id",
        "fetch_config_sha256",
        "fetch_pipeline_sha256",
        "manifest_request_sha256",
        "extractor_suite_version",
        "extractor_method",
        "extractor_version",
        "extractor_environment",
        "extractor_fingerprint_sha256",
        "robots",
        "failure_current_url",
        "error",
    ):
        updated[key] = document.get(key)
    previous_attempts = row.get("fetch_attempts", 0)
    if isinstance(previous_attempts, bool) or not isinstance(previous_attempts, int):
        previous_attempts = 0
    updated["fetch_attempts"] = previous_attempts + int(
        document.get("fetch_attempted") is True
    )
    return updated


def _load_retrieval_helpers() -> Tuple[Any, Any, Any, Any]:
    """Import shared IO helpers lazily to avoid an integration import cycle."""
    try:
        from factcheck_bench_retrieval import (  # type: ignore
            atomic_write_bytes,
            atomic_write_json,
            atomic_write_jsonl,
            load_jsonl,
        )
    except ImportError as exc:
        raise RuntimeError(
            "factcheck_bench_retrieval helpers are required: load_jsonl, "
            "atomic_write_jsonl, atomic_write_json, atomic_write_bytes"
        ) from exc
    return load_jsonl, atomic_write_jsonl, atomic_write_json, atomic_write_bytes


def _clear_extracted_document_fields(document: Dict[str, Any]) -> None:
    for key, value in (
        ("content_hash", None),
        ("clean_text", None),
        ("clean_text_sha256", None),
        ("clean_char_count", 0),
        ("content_blocks", []),
        ("title", None),
        ("text_encoding", None),
        ("page_count", None),
        ("html_extraction_source", None),
    ):
        document[key] = value


def _reprocess_markdown(report: Mapping[str, Any]) -> str:
    before = report["before_status_counts"]
    after = report["after_status_counts"]
    lines = [
        "# Frozen source-document reprocessing",
        "",
        f"- Status: **{report['status']}**",
        f"- Network used: **{report['network_was_used']}**",
        f"- Input document records: **{report['document_record_count']}**",
        f"- HTML documents reparsed: **{report['html_reparsed_count']}**",
        f"- Former empty-content documents recovered: **{report['recovered_empty_content_count']}**",
        f"- Former successes rejected as insufficient: **{report['downgraded_success_count']}**",
        f"- Successful document records after reprocessing: **{report['successful_document_record_count']}**",
        f"- Primary documents after deduplication: **{report['primary_document_count_after_deduplication']}**",
        f"- Projected passages at configured chunking: **{report['projected_passage_count_at_configured_chunking']}**",
        f"- Snapshot: `{report.get('snapshot_directory')}`",
        "",
        "| Fetch/extraction status | Before | After |",
        "|---|---:|---:|",
    ]
    for status in sorted(set(before) | set(after)):
        lines.append(f"| {status} | {before.get(status, 0)} | {after.get(status, 0)} |")
    lines.extend(
        [
            "",
            "The frozen raw bytes were reused; no URL was contacted. Canonical passages",
            "and dev qrels must be rebuilt after this stage.",
            "",
        ]
    )
    return "\n".join(lines)


def reprocess_frozen_documents(
    project_root: Path,
    paths: Any,
    config: Mapping[str, Any],
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Re-extract frozen HTML offline and migrate resume provenance safely.

    The first mutating run snapshots the current documents, URL manifest,
    passages, and associated reports under a content-addressed directory.
    Raw response bytes are never modified and no network operation occurs.
    """
    from factcheck_bench_retrieval import (  # type: ignore
        atomic_write_bytes,
        atomic_write_json,
        atomic_write_jsonl,
        atomic_write_text,
        canonical_json_hash,
        chunk_document,
        load_json,
        load_jsonl,
        sha256_file,
    )

    project_root = Path(project_root).resolve()
    documents_path = Path(paths.documents)
    manifest_path = Path(paths.url_manifest)
    if not documents_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Frozen documents and URL manifest are required before offline reprocessing"
        )
    documents = load_jsonl(documents_path)
    manifest_rows = load_jsonl(manifest_path)
    if len(documents) != len(manifest_rows):
        raise ValueError(
            "documents.jsonl and url_manifest.jsonl must retain one trace row per URL"
        )
    document_ids = [str(row.get("doc_id") or "") for row in documents]
    if any(not value for value in document_ids) or len(document_ids) != len(set(document_ids)):
        raise ValueError("Frozen document records require unique doc_id values")
    documents_by_manifest_id = {
        str(row.get("source_url_manifest_id")): row for row in documents
    }
    identities = [_manifest_identity(row, index) for index, row in enumerate(manifest_rows)]
    if set(documents_by_manifest_id) != set(identities):
        raise ValueError("Frozen documents do not match current URL manifest identities")

    generated_at = _utc_now()
    fetch_config_sha256 = _canonical_json_sha256(dict(_config_section(config)))
    fetch_pipeline_sha256 = _fetch_pipeline_sha256(fetch_config_sha256)
    input_documents_sha256 = sha256_file(documents_path)
    snapshot_directory = Path(paths.root) / "snapshots" / (
        "pre_reprocess_" + input_documents_sha256[:16]
    )
    before_counts = Counter(str(row.get("fetch_status")) for row in documents)
    new_documents: List[Dict[str, Any]] = []
    html_reparsed = 0
    recovered_empty = 0
    downgraded_success = 0
    raw_integrity_failures = 0
    changed = 0

    for index, original in enumerate(documents, start=1):
        document = dict(original)
        previous_status = str(document.get("fetch_status"))
        metadata_current = (
            document.get("extractor_suite_version") == EXTRACTOR_SUITE_VERSION
            and document.get("fetch_pipeline_sha256") == fetch_pipeline_sha256
            and document.get("fetch_config_sha256") == fetch_config_sha256
        )
        raw_path = _resolve_raw_path(document.get("raw_path"), project_root)
        content_type = str(document.get("content_type") or "").casefold()
        is_html_candidate = (
            previous_status in {"success", "empty_content"}
            and raw_path is not None
            and (
                "html" in content_type
                or str(document.get("document_type") or "").casefold() == "html"
                or raw_path.suffix.casefold() in {".html", ".htm"}
            )
        )
        html_current = (
            previous_status != "success"
            or (
                document.get("document_type") == "html"
                and document.get("extractor_version") == HTML_EXTRACTOR_VERSION
                and document.get("extractor_method")
                == "stdlib_htmlparser_readable_blocks_jsonld_fallback"
            )
        )
        if metadata_current and (not is_html_candidate or html_current):
            new_documents.append(document)
            print(
                f"[{index}/{len(documents)}] unchanged {previous_status}",
                flush=True,
            )
            continue

        changed += 1
        document["fetch_config_sha256"] = fetch_config_sha256
        document["fetch_pipeline_sha256"] = fetch_pipeline_sha256
        document["extractor_suite_version"] = EXTRACTOR_SUITE_VERSION
        document["fetch_attempted"] = False
        action = "metadata migrated"
        if is_html_candidate:
            html_reparsed += 1
            expected_raw_hash = document.get("raw_sha256") or document.get(
                "raw_content_sha256"
            )
            try:
                if raw_path is None:
                    raise OSError("raw path is missing")
                raw_path.resolve().relative_to(project_root)
                raw_payload = raw_path.read_bytes()
            except (OSError, ValueError) as exc:
                raw_payload = b""
                integrity_error = str(exc)
            else:
                actual_hash = _sha256_bytes(raw_payload)
                integrity_error = (
                    None
                    if isinstance(expected_raw_hash, str)
                    and actual_hash == expected_raw_hash
                    else "raw SHA-256 mismatch"
                )
            if integrity_error is not None:
                raw_integrity_failures += 1
                document["fetch_status"] = "raw_integrity_error"
                _clear_extracted_document_fields(document)
                document["error"] = {
                    "category": "raw_integrity_error",
                    "message": integrity_error,
                    "details": {},
                }
                action = "raw_integrity_error"
            else:
                try:
                    parsed = _parse_html(raw_payload, str(document.get("content_type") or "text/html"))
                except FetchFailure as exc:
                    document["fetch_status"] = exc.category
                    _clear_extracted_document_fields(document)
                    document.update(
                        {
                            "document_type": "html",
                            "extractor_method": (
                                "stdlib_htmlparser_readable_blocks_jsonld_fallback"
                            ),
                            "extractor_version": HTML_EXTRACTOR_VERSION,
                            "extractor_environment": _extractor_registry()["html"],
                            "extractor_fingerprint_sha256": _extractor_fingerprint(
                                "html",
                                "stdlib_htmlparser_readable_blocks_jsonld_fallback",
                                HTML_EXTRACTOR_VERSION,
                            ),
                            "error": {
                                "category": exc.category,
                                "message": exc.message,
                                "details": exc.details,
                            },
                        }
                    )
                    if previous_status == "success":
                        downgraded_success += 1
                    action = exc.category
                else:
                    clean_text = parsed["clean_text"]
                    clean_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
                    document.update(parsed)
                    document.update(
                        {
                            "fetch_status": "success",
                            "clean_text_sha256": clean_hash,
                            "content_hash": clean_hash,
                            "clean_char_count": len(clean_text),
                            "extractor_environment": _extractor_registry()["html"],
                            "extractor_fingerprint_sha256": _extractor_fingerprint(
                                "html",
                                str(parsed["extractor_method"]),
                                str(parsed["extractor_version"]),
                            ),
                            "error": None,
                        }
                    )
                    if previous_status == "empty_content":
                        recovered_empty += 1
                    action = (
                        "recovered"
                        if previous_status == "empty_content"
                        else "reparsed"
                    )
        elif previous_status == "success":
            document_type = str(document.get("document_type") or "")
            registry = _extractor_registry().get(document_type)
            if registry is None:
                raise ValueError(
                    f"Successful document has unsupported type during migration: {document_type}"
                )
            document["extractor_environment"] = registry
            document["extractor_fingerprint_sha256"] = _extractor_fingerprint(
                document_type,
                str(document.get("extractor_method")),
                str(document.get("extractor_version")),
            )
        document["offline_reprocess"] = {
            "version": "frozen_html_reprocess_v1",
            "reprocessed_at": generated_at,
            "source_raw_sha256": document.get("raw_sha256"),
            "network_used": False,
        }
        new_documents.append(document)
        print(
            f"[{index}/{len(documents)}] {action}: {document.get('doc_id')}",
            flush=True,
        )

    _assign_duplicate_groups(new_documents)
    successful_documents = [
        row
        for row in new_documents
        if row.get("fetch_status") == "success"
        and isinstance(row.get("clean_text"), str)
        and row["clean_text"].strip()
    ]
    primary_documents = [
        row
        for row in successful_documents
        if not row.get("duplicate_of_doc_id")
        and row.get("orphaned_from_url_manifest") is not True
    ]
    chunk_config = config["chunking"]
    chunk_spec = {
        "chunking_version": chunk_config["version"],
        "tokenizer": chunk_config["tokenizer"],
        "chunk_size_tokens": int(chunk_config["chunk_size_tokens"]),
        "chunk_overlap_tokens": int(chunk_config["chunk_overlap_tokens"]),
        "minimum_boundary_fraction": float(
            chunk_config["minimum_boundary_fraction"]
        ),
    }
    chunk_fingerprint = canonical_json_hash(chunk_spec)
    projected_passage_count = sum(
        len(
            chunk_document(
                document,
                chunk_size=chunk_spec["chunk_size_tokens"],
                chunk_overlap=chunk_spec["chunk_overlap_tokens"],
                minimum_boundary_fraction=chunk_spec[
                    "minimum_boundary_fraction"
                ],
                chunking_version=chunk_spec["chunking_version"],
                chunk_config_fingerprint=chunk_fingerprint,
            )
        )
        for document in primary_documents
    )
    by_manifest_id = {
        str(row.get("source_url_manifest_id")): row for row in new_documents
    }
    new_manifest = [
        _manifest_fetch_update(row, by_manifest_id[identities[index]])
        for index, row in enumerate(manifest_rows)
    ]
    after_counts = Counter(str(row.get("fetch_status")) for row in new_documents)
    report: Dict[str, Any] = {
        "schema_version": FETCH_SCHEMA_VERSION,
        "status": "dry_run" if dry_run else "complete",
        "generated_at": generated_at,
        "network_was_used": False,
        "document_record_count": len(documents),
        "changed_document_count": changed,
        "html_reparsed_count": html_reparsed,
        "recovered_empty_content_count": recovered_empty,
        "downgraded_success_count": downgraded_success,
        "raw_integrity_failure_count": raw_integrity_failures,
        "successful_document_record_count": len(successful_documents),
        "primary_document_count_after_deduplication": len(primary_documents),
        "projected_passage_count_at_configured_chunking": projected_passage_count,
        "projected_chunking": {
            **chunk_spec,
            "fingerprint": chunk_fingerprint,
        },
        "before_status_counts": dict(sorted(before_counts.items())),
        "after_status_counts": dict(sorted(after_counts.items())),
        "input_documents_sha256": input_documents_sha256,
        "fetch_config_sha256": fetch_config_sha256,
        "fetch_pipeline_sha256": fetch_pipeline_sha256,
        "extractor_suite_version": EXTRACTOR_SUITE_VERSION,
        "html_extractor_version": HTML_EXTRACTOR_VERSION,
        "minimum_html_clean_chars": _MIN_HTML_CLEAN_CHARS,
        "minimum_html_words": _MIN_HTML_WORDS,
        "snapshot_directory": _project_relative(snapshot_directory, project_root),
        "downstream_status": "passages_and_dev_qrels_require_rebuild",
    }
    if dry_run:
        return report
    if changed == 0:
        prior_report = load_json(Path(paths.reprocess_report_json), allow_missing=True)
        if (
            prior_report.get("output_documents_sha256") == input_documents_sha256
            and prior_report.get("extractor_suite_version")
            == EXTRACTOR_SUITE_VERSION
        ):
            result = dict(prior_report)
            result["status"] = "already_current"
            result["changed_document_count"] = 0
            return result

    snapshot_assets = [
        documents_path,
        manifest_path,
        Path(paths.passages),
        Path(paths.passage_build_report),
        Path(paths.fetch_report),
    ]
    for attribute in (
        "qrels_jsonl",
        "qrels_tsv",
        "qrels_mapping_audit",
        "qrels_mapping_report_json",
        "qrels_mapping_report_markdown",
        "qrels_dev_jsonl",
        "qrels_dev_tsv",
        "qrels_dev_mapping_audit",
        "qrels_dev_mapping_report_json",
        "qrels_dev_mapping_report_markdown",
        "qrels_heldout_jsonl",
        "qrels_heldout_tsv",
        "qrels_heldout_mapping_audit",
        "qrels_heldout_mapping_report_json",
        "qrels_heldout_mapping_report_markdown",
        "corpus_summary_json",
        "corpus_summary_markdown",
    ):
        candidate = getattr(paths, attribute, None)
        if candidate is not None:
            snapshot_assets.append(Path(candidate))
    snapshot_manifest: Dict[str, Any] = {
        "schema_version": "fcb_pre_reprocess_snapshot_v1",
        "created_at": generated_at,
        "source_documents_sha256": input_documents_sha256,
        "files": [],
    }
    for source in snapshot_assets:
        if not source.is_file():
            continue
        destination = snapshot_directory / source.name
        if not destination.exists():
            atomic_write_bytes(destination, source.read_bytes())
        snapshot_manifest["files"].append(
            {
                "source": _project_relative(source, project_root),
                "snapshot": _project_relative(destination, project_root),
                "sha256": sha256_file(source),
            }
        )
    atomic_write_json(snapshot_directory / "snapshot_manifest.json", snapshot_manifest)
    atomic_write_jsonl(documents_path, new_documents)
    atomic_write_jsonl(manifest_path, new_manifest)
    report["output_documents_sha256"] = sha256_file(documents_path)
    report["output_url_manifest_sha256"] = sha256_file(manifest_path)
    atomic_write_json(Path(paths.reprocess_report_json), report)
    atomic_write_text(
        Path(paths.reprocess_report_markdown),
        _reprocess_markdown(report),
    )
    return report


def fetch_corpus(
    project_root: Path,
    paths: Any,
    config: Mapping[str, Any],
    *,
    limit: Optional[int] = None,
    resume: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Fetch, freeze, clean, and audit source documents from ``url_manifest``.

    ``limit`` is the maximum number of non-resumed manifest rows attempted in
    this invocation.  Successfully resumed rows do not consume the limit.
    ``dry_run`` performs no DNS/network access and writes no artifacts.

    Existing successful rows are skipped only when the raw file exists and its
    SHA-256 still matches the persisted hash. Completed failures are also
    skipped under resume; stale/interrupted rows are retried, while explicit
    ``resume=False`` (the CLI's ``--refetch``) retries prior outcomes.
    Duplicate canonical URLs reuse an already verified success during the same
    run (or a prior run) but retain a separate document/manifest trace row.
    """
    project_root = Path(project_root).resolve()
    settings = _settings_from_config(config)
    fetch_config_sha256 = _canonical_json_sha256(dict(_config_section(config)))
    fetch_pipeline_sha256 = _fetch_pipeline_sha256(fetch_config_sha256)
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer or None")

    (
        load_jsonl,
        atomic_write_jsonl,
        atomic_write_json,
        atomic_write_bytes,
    ) = _load_retrieval_helpers()
    manifest_path = Path(paths.url_manifest)
    documents_path = Path(paths.documents)
    raw_documents_dir = Path(paths.raw_documents_dir)
    fetch_report_path_value = getattr(paths, "fetch_report", None)
    fetch_report_path = (
        Path(fetch_report_path_value) if fetch_report_path_value is not None else None
    )
    if not manifest_path.is_file():
        raise FileNotFoundError("URL manifest does not exist: %s" % manifest_path)

    manifest_rows = load_jsonl(manifest_path)
    if not isinstance(manifest_rows, list) or any(
        not isinstance(row, Mapping) for row in manifest_rows
    ):
        raise ValueError("url_manifest must contain one JSON object per line")
    identities = [_manifest_identity(row, index) for index, row in enumerate(manifest_rows)]
    duplicates = [item for item, count in Counter(identities).items() if count > 1]
    if duplicates:
        raise ValueError("url_manifest identities are not unique: %s" % duplicates[:5])

    existing_documents = load_jsonl(documents_path) if documents_path.is_file() else []
    existing_manifest_ids = [
        str(row.get("source_url_manifest_id"))
        for row in existing_documents
        if isinstance(row, Mapping) and row.get("source_url_manifest_id")
    ]
    duplicate_existing_manifest_ids = [
        item
        for item, count in Counter(existing_manifest_ids).items()
        if count > 1
    ]
    if duplicate_existing_manifest_ids:
        raise ValueError(
            "documents.jsonl contains duplicate source_url_manifest_id values: %s"
            % duplicate_existing_manifest_ids[:5]
        )
    existing_by_manifest_id = {
        str(row.get("source_url_manifest_id")): row
        for row in existing_documents
        if isinstance(row, Mapping) and row.get("source_url_manifest_id")
    }
    existing_orphans = [
        dict(row)
        for row in existing_documents
        if isinstance(row, Mapping)
        and str(row.get("source_url_manifest_id")) not in set(identities)
    ]

    report: Dict[str, Any] = {
        "schema_version": FETCH_SCHEMA_VERSION,
        "corpus_name": (
            config.get("corpus", {}).get("name")
            if isinstance(config.get("corpus"), Mapping)
            else "benchmark-grounded closed source-document corpus"
        ),
        "started_at": _utc_now(),
        "dry_run": dry_run,
        "resume": resume,
        "limit": limit,
        "url_manifest_path": _project_relative(manifest_path, project_root),
        "documents_path": _project_relative(documents_path, project_root),
        "raw_documents_dir": _project_relative(raw_documents_dir, project_root),
        "fetch_report_path": (
            _project_relative(fetch_report_path, project_root)
            if fetch_report_path is not None
            else None
        ),
        "manifest_row_count": len(manifest_rows),
        "existing_document_count": len(existing_documents),
        "orphan_existing_document_count": len(existing_orphans),
        "attempted_count": 0,
        "resumed_success_count": 0,
        "resumed_failure_count": 0,
        "canonical_reuse_count": 0,
        "unprocessed_due_to_limit_count": 0,
        "status_counts": {},
        "network_was_used": False,
        "writes_performed": False,
        "fetch_config_sha256": fetch_config_sha256,
        "fetch_pipeline_sha256": fetch_pipeline_sha256,
        "extractor_suite_version": EXTRACTOR_SUITE_VERSION,
        "checkpoint_every": settings.checkpoint_every,
        "url_wall_clock_timeout_seconds": settings.url_wall_clock_timeout_seconds,
        "periodic_checkpoint_count": 0,
        "last_checkpoint_at": None,
        "last_checkpoint_attempted_count": 0,
    }

    def resume_valid(
        previous: Mapping[str, Any], row: Mapping[str, Any], url: Optional[str]
    ) -> bool:
        return _verified_resume_record(
            previous,
            project_root,
            manifest_row=row,
            requested_url=url,
            fetch_config_sha256=fetch_config_sha256,
            fetch_pipeline_sha256=fetch_pipeline_sha256,
        )

    def resume_failure_valid(
        previous: Mapping[str, Any], row: Mapping[str, Any], url: Optional[str]
    ) -> bool:
        return _verified_resume_failure_record(
            previous,
            project_root,
            manifest_row=row,
            requested_url=url,
            fetch_config_sha256=fetch_config_sha256,
            fetch_pipeline_sha256=fetch_pipeline_sha256,
        )

    if dry_run:
        syntax_counts: Counter = Counter()
        pending = 0
        for index, row in enumerate(manifest_rows):
            identity = identities[index]
            previous = existing_by_manifest_id.get(identity)
            url = _url_from_manifest(row)
            if resume and previous is not None and resume_valid(previous, row, url):
                syntax_counts["verified_resume_success"] += 1
                continue
            if (
                resume
                and previous is not None
                and resume_failure_valid(previous, row, url)
            ):
                syntax_counts["verified_resume_failure"] += 1
                continue
            if limit is not None and pending >= limit:
                syntax_counts["not_inspected_due_to_limit"] += 1
                continue
            pending += 1
            if url is None:
                if row.get("canonicalisation_status") not in (None, "ok"):
                    syntax_counts["canonicalisation_invalid"] += 1
                else:
                    syntax_counts["missing_url"] += 1
                continue
            try:
                _basic_normalise_url(url)
            except FetchFailure as exc:
                syntax_counts[exc.category] += 1
            else:
                syntax_counts["ready_for_network_fetch"] += 1
        report.update(
            {
                "planned_attempt_count": pending,
                "status_counts": dict(sorted(syntax_counts.items())),
                "finished_at": _utc_now(),
            }
        )
        return report

    progress_total = len(manifest_rows)
    print(
        "Fetch corpus: "
        f"{progress_total} URL rows; resume={resume}; "
        f"existing={len(existing_documents)}; "
        f"limit={limit if limit is not None else 'none'}; "
        f"checkpoint_every={settings.checkpoint_every}; "
        f"url_timeout={settings.url_wall_clock_timeout_seconds:g}s",
        flush=True,
    )

    raw_documents_dir.mkdir(parents=True, exist_ok=True)
    robots_cache: Dict[str, Dict[str, Any]] = {}
    robots_delay_by_origin: Dict[str, float] = {}
    last_request_by_origin: Dict[str, float] = {}
    successful_by_canonical: Dict[str, Dict[str, Any]] = {}
    documents_by_manifest_id: Dict[str, Dict[str, Any]] = {}

    manifest_by_identity = dict(zip(identities, manifest_rows))
    for identity, previous in existing_by_manifest_id.items():
        persisted = dict(previous)
        # This flag is invocation-local.  Persisted attempt counts already
        # include earlier runs and must not be incremented by a checkpoint that
        # happens before this row is revisited.
        persisted["fetch_attempted"] = False
        persisted.pop("orphaned_from_url_manifest", None)
        persisted.pop("source_trace", None)
        persisted["evaluation_only_source_trace"] = _evaluation_only_trace(
            manifest_by_identity[identity]
        )
        documents_by_manifest_id[identity] = persisted

    def write_checkpoint() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Atomically preserve all completed rows without periodic O(N^2) writes."""
        checkpoint_documents = [
            documents_by_manifest_id[identity]
            for identity in identities
            if identity in documents_by_manifest_id
        ]
        for existing_orphan in existing_orphans:
            orphan = dict(existing_orphan)
            legacy_trace = orphan.pop("source_trace", None)
            existing_evaluation_trace = orphan.pop(
                "evaluation_only_source_trace", None
            )
            trace_payload = None
            if isinstance(existing_evaluation_trace, Mapping) and isinstance(
                existing_evaluation_trace.get("trace"), Mapping
            ):
                trace_payload = existing_evaluation_trace["trace"]
            elif isinstance(legacy_trace, Mapping):
                trace_payload = legacy_trace
            if trace_payload is not None:
                orphan["evaluation_only_source_trace"] = _evaluation_only_trace(
                    {"trace": trace_payload}
                )
            if orphan.get("fetch_status") == "success":
                # Orphans remain auditable but are not active corpus sources;
                # passage construction selects only fetch_status=success.
                orphan["orphaned_original_fetch_status"] = "success"
                orphan["fetch_status"] = "orphaned_success"
            orphan["orphaned_from_url_manifest"] = True
            checkpoint_documents.append(orphan)
        _assign_duplicate_groups(checkpoint_documents)
        checkpoint_by_id = {
            str(document.get("source_url_manifest_id")): document
            for document in checkpoint_documents
            if document.get("source_url_manifest_id")
        }
        checkpoint_manifest = []
        for manifest_index, manifest_row in enumerate(manifest_rows):
            current = checkpoint_by_id.get(identities[manifest_index])
            checkpoint_manifest.append(
                _manifest_fetch_update(manifest_row, current)
                if current is not None
                else dict(manifest_row)
            )
        atomic_write_jsonl(documents_path, checkpoint_documents)
        atomic_write_jsonl(manifest_path, checkpoint_manifest)
        return checkpoint_documents, checkpoint_manifest

    if resume:
        verified_by_canonical: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
        for index, row in enumerate(manifest_rows):
            identity = identities[index]
            previous = existing_by_manifest_id.get(identity)
            url = _url_from_manifest(row)
            if previous is None or not resume_valid(previous, row, url):
                continue
            if url is None:
                continue
            try:
                canonical_key = _basic_normalise_url(url)
            except FetchFailure:
                continue
            verified_by_canonical[canonical_key].append(dict(previous))
        for canonical_key, candidates in verified_by_canonical.items():
            candidates.sort(
                key=lambda candidate: (
                    candidate.get("fetch_reused_from_doc_id") is not None,
                    str(candidate.get("doc_id", "")),
                )
            )
            successful_by_canonical[canonical_key] = candidates[0]

    attempted = 0
    try:
        for index, row in enumerate(manifest_rows):
            progress_position = index + 1
            identity = identities[index]
            url = _url_from_manifest(row)
            doc_id = _doc_id(row, identity)
            previous = existing_by_manifest_id.get(identity)

            if resume and previous is not None and resume_valid(previous, row, url):
                document = dict(previous)
                document.pop("source_trace", None)
                document["evaluation_only_source_trace"] = _evaluation_only_trace(row)
                document.pop("orphaned_from_url_manifest", None)
                document["fetch_attempted"] = False
                report["resumed_success_count"] += 1
                _print_fetch_progress(
                    progress_position,
                    progress_total,
                    "skipped (verified resume success), "
                    + _format_progress_bytes(document.get("raw_byte_count")),
                )
            elif (
                resume
                and previous is not None
                and resume_failure_valid(previous, row, url)
            ):
                document = dict(previous)
                document.pop("source_trace", None)
                document["evaluation_only_source_trace"] = _evaluation_only_trace(row)
                document.pop("orphaned_from_url_manifest", None)
                document["fetch_attempted"] = False
                report["resumed_failure_count"] += 1
                _print_fetch_progress(
                    progress_position,
                    progress_total,
                    "skipped (recorded failure: "
                    f"{document.get('fetch_status')}), use --refetch to retry",
                )
            elif limit is not None and attempted >= limit:
                if previous is not None:
                    document = dict(previous)
                    document["fetch_attempted"] = False
                    if document.get("fetch_status") == "success" and not resume_valid(
                        previous, row, url
                    ):
                        document["fetch_status"] = "stale_pending"
                        document["error"] = {
                            "category": "stale_pending",
                            "message": (
                                "Prior success did not match the current manifest/config/"
                                "extractor fingerprint and was not refreshed due to limit"
                            ),
                            "details": {},
                        }
                else:
                    document = _base_document(
                        row,
                        identity,
                        doc_id,
                        url,
                        fetch_config_sha256,
                        fetch_pipeline_sha256,
                    )
                report["unprocessed_due_to_limit_count"] += 1
            else:
                attempted += 1
                canonical_key = None
                if url is not None:
                    try:
                        canonical_key = _basic_normalise_url(url)
                    except FetchFailure:
                        canonical_key = url
                reusable = successful_by_canonical.get(canonical_key or "")
                if reusable is not None:
                    _print_fetch_progress(
                        progress_position,
                        progress_total,
                        f"reusing {_progress_target(url)} ...",
                    )
                    base = _base_document(
                        row,
                        identity,
                        doc_id,
                        url,
                        fetch_config_sha256,
                        fetch_pipeline_sha256,
                    )
                    document = _clone_success(reusable, base)
                    report["canonical_reuse_count"] += 1
                    _print_fetch_progress(
                        progress_position,
                        progress_total,
                        _fetch_result_message(document, reused=True),
                    )
                else:
                    if url is not None:
                        try:
                            _basic_normalise_url(url)
                        except FetchFailure:
                            pass
                        else:
                            report["network_was_used"] = True
                    _print_fetch_progress(
                        progress_position,
                        progress_total,
                        f"fetching {_progress_target(url)} ...",
                    )
                    try:
                        with _url_wall_clock_timeout(
                            settings.url_wall_clock_timeout_seconds
                        ):
                            document = _fetch_one(
                                row,
                                identity,
                                doc_id,
                                url,
                                project_root,
                                raw_documents_dir,
                                settings,
                                robots_cache,
                                robots_delay_by_origin,
                                last_request_by_origin,
                                atomic_write_bytes,
                                fetch_config_sha256,
                                fetch_pipeline_sha256,
                            )
                    except URLWallClockTimeout as exc:
                        document = _base_document(
                            row,
                            identity,
                            doc_id,
                            url,
                            fetch_config_sha256,
                            fetch_pipeline_sha256,
                        )
                        document.update(
                            {
                                "fetched_at": _utc_now(),
                                "fetch_attempted": True,
                                "fetch_status": "url_timeout",
                                "error": {
                                    "category": "url_timeout",
                                    "message": str(exc),
                                    "details": {
                                        "wall_clock_timeout_seconds": (
                                            settings.url_wall_clock_timeout_seconds
                                        )
                                    },
                                },
                            }
                        )
                    _print_fetch_progress(
                        progress_position,
                        progress_total,
                        _fetch_result_message(document),
                    )
                    if document.get("fetch_status") == "success" and canonical_key:
                        successful_by_canonical.setdefault(canonical_key, document)
            documents_by_manifest_id[identity] = document
            if (
                attempted > 0
                and attempted % settings.checkpoint_every == 0
                and attempted != report["last_checkpoint_attempted_count"]
            ):
                checkpoint_documents, _ = write_checkpoint()
                checkpoint_at = _utc_now()
                report.update(
                    {
                        "status": "running_checkpoint",
                        "attempted_count": attempted,
                        "document_count": len(checkpoint_documents),
                        "writes_performed": True,
                        "periodic_checkpoint_count": (
                            int(report["periodic_checkpoint_count"]) + 1
                        ),
                        "last_checkpoint_at": checkpoint_at,
                        "last_checkpoint_attempted_count": attempted,
                    }
                )
                if fetch_report_path is not None:
                    atomic_write_json(fetch_report_path, report)
                print(
                    f"[checkpoint] saved after {attempted} attempted URL rows "
                    f"({progress_position}/{progress_total} manifest rows visited)",
                    flush=True,
                )
    except BaseException as error:
        # KeyboardInterrupt/SystemExit are intentionally included: preserve the
        # latest completed row, then re-raise the original interruption.
        try:
            checkpoint_documents, _ = write_checkpoint()
            report.update(
                {
                    "status": "interrupted_checkpoint",
                    "attempted_count": attempted,
                    "document_count": len(checkpoint_documents),
                    "writes_performed": True,
                    "last_checkpoint_at": _utc_now(),
                    "last_checkpoint_attempted_count": attempted,
                    "interruption": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                    "finished_at": _utc_now(),
                }
            )
            if fetch_report_path is not None:
                atomic_write_json(fetch_report_path, report)
            print(
                f"[interrupted] checkpoint saved after {attempted} attempted URL rows; "
                "restart with --resume to continue",
                flush=True,
            )
        except Exception:
            LOGGER.exception("Failed to persist corpus checkpoint during interruption")
        raise

    documents, _ = write_checkpoint()
    status_counts = Counter(
        str(document.get("fetch_status"))
        for document in documents
        if not document.get("orphaned_from_url_manifest")
    )
    report.update(
        {
            "status": "partial_limit" if limit is not None else "complete",
            "attempted_count": attempted,
            "document_count": len(documents),
            "active_document_count": len(documents) - len(existing_orphans),
            "success_count": status_counts.get("success", 0),
            "status_counts": dict(sorted(status_counts.items())),
            "duplicate_document_count": sum(bool(row.get("is_duplicate")) for row in documents),
            "duplicate_component_count": len(
                {
                    row.get("duplicate_component_id")
                    for row in documents
                    if row.get("duplicate_component_id")
                }
            ),
            "writes_performed": True,
            "final_checkpoint_at": _utc_now(),
            "finished_at": _utc_now(),
        }
    )
    if fetch_report_path is not None:
        atomic_write_json(fetch_report_path, report)
    print(
        "Fetch corpus complete: "
        f"attempted={attempted}, "
        f"resumed={report['resumed_success_count']}, "
        f"failed_skips={report['resumed_failure_count']}, "
        f"reused={report['canonical_reuse_count']}, "
        f"statuses={dict(sorted(status_counts.items()))}",
        flush=True,
    )
    return report


__all__ = ["fetch_corpus"]
