# Validated HTTPS access for CFB provider requests.

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
)


class UrlSecurityError(ValueError):
    """Raised when a provider URL violates the outbound request policy."""


def _host_allowlist(
    allowed_hosts: Iterable[str],
) -> frozenset[str]:
    hosts = frozenset(
        str(host).strip().casefold()
        for host in allowed_hosts
        if str(host).strip()
    )

    if not hosts:
        raise ValueError(
            "allowed_hosts must contain at least one hostname"
        )

    return hosts


def validate_https_url(
    url: str,
    *,
    allowed_hosts: Iterable[str],
    label: str = "URL",
) -> str:
    text = str(
        url or ""
    ).strip()

    if not text:
        raise UrlSecurityError(
            f"{label} is blank"
        )

    try:
        parsed = urlsplit(
            text
        )
        port = parsed.port
    except ValueError as exc:
        raise UrlSecurityError(
            f"{label} is malformed: {text!r}"
        ) from exc

    if parsed.scheme.casefold() != "https":
        raise UrlSecurityError(
            f"{label} must use HTTPS: {text!r}"
        )

    hostname = (
        parsed.hostname
        or ""
    ).casefold()

    hosts = _host_allowlist(
        allowed_hosts
    )

    if hostname not in hosts:
        raise UrlSecurityError(
            f"{label} has unapproved host: {text!r}"
        )

    if (
        parsed.username is not None
        or parsed.password is not None
    ):
        raise UrlSecurityError(
            f"{label} must not contain URL credentials: {text!r}"
        )

    if port not in {
        None,
        443,
    }:
        raise UrlSecurityError(
            f"{label} has unapproved port: {text!r}"
        )

    return text


class _AllowlistedRedirectHandler(
    HTTPRedirectHandler
):
    def __init__(
        self,
        allowed_hosts: Iterable[str],
    ) -> None:
        super().__init__()
        self._allowed_hosts = _host_allowlist(
            allowed_hosts
        )

    def redirect_request(
        self,
        req,
        fp,
        code,
        msg,
        headers,
        newurl,
    ):
        safe_url = validate_https_url(
            newurl,
            allowed_hosts=self._allowed_hosts,
            label="redirect URL",
        )

        return super().redirect_request(
            req,
            fp,
            code,
            msg,
            headers,
            safe_url,
        )


def open_https(
    request: Request | str,
    *,
    allowed_hosts: Iterable[str],
    timeout: float,
):
    hosts = _host_allowlist(
        allowed_hosts
    )

    request_url = (
        request.full_url
        if isinstance(
            request,
            Request,
        )
        else str(
            request
        )
    )

    validate_https_url(
        request_url,
        allowed_hosts=hosts,
        label="request URL",
    )

    opener = build_opener(
        _AllowlistedRedirectHandler(
            hosts
        )
    )

    return opener.open(
        request,
        timeout=timeout,
    )
