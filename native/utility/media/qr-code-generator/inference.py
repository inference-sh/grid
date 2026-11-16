import base64
import io
import logging
import math
import mimetypes
import os
import re
import tempfile
from datetime import datetime, timezone
from typing import List, Literal, Optional
from urllib.parse import quote
from xml.sax.saxutils import escape

import cairosvg
import segno
import segno.helpers as qrhelpers
from PIL import Image
from pydantic import Field, model_validator

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, ImageMeta, OutputMeta

COLOR = r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$"
COLOR_OR_TRANSPARENT = r"^(#([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})|transparent)$"
ISO_DATETIME = r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?(Z|[+-]\d{2}:?\d{2})?$"

ContentType = Literal[
    "url", "text", "email", "phone", "sms", "whatsapp", "wifi",
    "vcard", "mecard", "geo", "event", "epc_payment",
]
ModuleStyle = Literal[
    "square", "rounded", "dots", "diamond", "fluid", "leaf",
    "small-squares", "horizontal-bars", "vertical-bars",
]
EyeFrameStyle = Literal["square", "rounded", "extra-rounded", "circle", "leaf"]
EyeBallStyle = Literal["square", "rounded", "circle", "diamond", "leaf"]
OutputFormat = Literal["png", "svg", "pdf", "jpg", "webp", "eps"]


class AppInput(BaseAppInput):
    # ---- content ----
    content_type: ContentType = Field(
        default="url",
        description="What the QR code holds. url/text/phone/sms/whatsapp/email use `data`; wifi, vcard, mecard, geo, event and epc_payment use their own fields.",
    )
    data: Optional[str] = Field(
        default=None, max_length=4000,
        description="Main value: the URL or text, the phone number (phone, sms, whatsapp), or the recipient address (email).",
    )
    subject: Optional[str] = Field(default=None, max_length=300, description="Email subject (email).")
    message: Optional[str] = Field(default=None, max_length=2000, description="Message body (email, sms, whatsapp).")

    wifi_ssid: Optional[str] = Field(default=None, max_length=64, description="Wi-Fi network name (wifi).")
    wifi_password: Optional[str] = Field(default=None, max_length=128, description="Wi-Fi password (wifi).")
    wifi_security: Literal["WPA", "WEP", "nopass"] = Field(default="WPA", description="Wi-Fi security (wifi). WPA covers WPA2/WPA3.")
    wifi_hidden: bool = Field(default=False, description="The Wi-Fi network is hidden (wifi).")

    contact_first_name: Optional[str] = Field(default=None, max_length=100, description="First name (vcard, mecard).")
    contact_last_name: Optional[str] = Field(default=None, max_length=100, description="Last name (vcard, mecard).")
    contact_organization: Optional[str] = Field(default=None, max_length=200, description="Company (vcard).")
    contact_title: Optional[str] = Field(default=None, max_length=200, description="Job title (vcard).")
    contact_phone: Optional[str] = Field(default=None, max_length=40, description="Phone number (vcard, mecard).")
    contact_email: Optional[str] = Field(default=None, max_length=200, description="Email (vcard, mecard).")
    contact_website: Optional[str] = Field(default=None, max_length=300, description="Website (vcard, mecard).")
    contact_street: Optional[str] = Field(default=None, max_length=200, description="Street address (vcard).")
    contact_city: Optional[str] = Field(default=None, max_length=100, description="City (vcard, mecard).")
    contact_zip: Optional[str] = Field(default=None, max_length=20, description="Postal code (vcard, mecard).")
    contact_country: Optional[str] = Field(default=None, max_length=100, description="Country (vcard, mecard).")
    contact_note: Optional[str] = Field(default=None, max_length=500, description="Note (vcard, mecard).")

    latitude: Optional[float] = Field(default=None, ge=-90, le=90, description="Latitude (geo).")
    longitude: Optional[float] = Field(default=None, ge=-180, le=180, description="Longitude (geo).")

    event_title: Optional[str] = Field(default=None, max_length=200, description="Event title (event).")
    event_start: Optional[str] = Field(default=None, pattern=ISO_DATETIME, description="Start, ISO 8601: 2025-06-01 (all day) or 2025-06-01T18:00 or with Z/offset (event).")
    event_end: Optional[str] = Field(default=None, pattern=ISO_DATETIME, description="End, ISO 8601 (event).")
    event_location: Optional[str] = Field(default=None, max_length=300, description="Event location (event).")
    event_description: Optional[str] = Field(default=None, max_length=1000, description="Event description (event).")

    payment_name: Optional[str] = Field(default=None, max_length=70, description="Beneficiary name (epc_payment, SEPA EUR transfer).")
    payment_iban: Optional[str] = Field(default=None, pattern=r"^[A-Za-z]{2}[0-9]{2}[A-Za-z0-9 ]{10,36}$", description="Beneficiary IBAN (epc_payment).")
    payment_bic: Optional[str] = Field(default=None, pattern=r"^[A-Za-z0-9]{8}([A-Za-z0-9]{3})?$", description="Beneficiary BIC (epc_payment, optional).")
    payment_amount: Optional[float] = Field(default=None, ge=0.01, le=999999999.99, description="Amount in EUR (epc_payment).")
    payment_reference: Optional[str] = Field(default=None, max_length=140, description="Remittance text shown to the payer (epc_payment).")

    # ---- encoding ----
    error_correction: Literal["auto", "L", "M", "Q", "H"] = Field(
        default="auto",
        description="Error correction: L 7%, M 15%, Q 25%, H 30%. auto picks M, or H when a logo is added (EPC payments are always M).",
    )
    qr_version: Optional[int] = Field(default=None, ge=1, le=40, description="Force a QR version (1-40, grid of 21 to 177 modules). Empty picks the smallest that fits.")

    # ---- style ----
    module_style: ModuleStyle = Field(default="square", description="Shape of the data dots.")
    eye_frame_style: EyeFrameStyle = Field(default="square", description="Shape of the outer ring of the three corner eyes.")
    eye_ball_style: EyeBallStyle = Field(default="square", description="Shape of the inner square of the three corner eyes.")
    foreground_color: str = Field(default="#000000", pattern=COLOR, description="Color of the code, hex #RGB, #RRGGBB or #RRGGBBAA.")
    background_color: str = Field(default="#FFFFFF", pattern=COLOR_OR_TRANSPARENT, description="Background color, hex, or 'transparent' (JPG falls back to white).")
    eye_frame_color: Optional[str] = Field(default=None, pattern=COLOR, description="Color of the eye rings. Empty uses the foreground color (or gradient).")
    eye_ball_color: Optional[str] = Field(default=None, pattern=COLOR, description="Color of the eye centers. Empty uses the foreground color (or gradient).")
    gradient_type: Literal["none", "linear", "radial"] = Field(default="none", description="Fill the code with a gradient instead of the foreground color.")
    gradient_start_color: str = Field(default="#000000", pattern=COLOR, description="Gradient start color (center for radial).")
    gradient_end_color: str = Field(default="#3B5BDB", pattern=COLOR, description="Gradient end color (edge for radial).")
    gradient_angle: int = Field(default=45, ge=0, le=360, description="Linear gradient angle in degrees (0 = left to right, 90 = top to bottom).")

    # ---- logo ----
    logo: Optional[File] = Field(default=None, description="Logo placed in the center: PNG, JPG, WEBP, GIF or SVG.")
    logo_size: float = Field(default=0.22, ge=0.05, le=0.3, description="Logo width as a fraction of the code width (max 0.3 keeps it scannable).")
    logo_margin: float = Field(default=1.0, ge=0, le=4, description="Empty space around the logo, in modules.")
    logo_shape: Literal["square", "rounded", "circle"] = Field(default="rounded", description="Shape of the area cleared behind the logo.")
    logo_background_color: Optional[str] = Field(default=None, pattern=COLOR, description="Fill the cleared logo area with this color. Empty shows the background.")

    # ---- frame and label ----
    frame_style: Literal["none", "border", "banner"] = Field(default="none", description="none, a rounded border, or a border with a filled label banner (like 'SCAN ME' frames).")
    frame_color: str = Field(default="#000000", pattern=COLOR, description="Border and banner color.")
    label_text: Optional[str] = Field(default=None, max_length=60, description="Caption such as 'SCAN ME'.")
    label_position: Literal["bottom", "top"] = Field(default="bottom", description="Where the caption goes.")
    label_color: Optional[str] = Field(default=None, pattern=COLOR, description="Caption color. Empty: foreground color, or the background color on a banner.")

    # ---- output ----
    output_formats: List[OutputFormat] = Field(
        default_factory=lambda: ["png", "svg"], min_length=1, max_length=6,
        description="Files to produce. svg, pdf and eps are vector (scale to any size); png, jpg and webp are raster at `size` pixels.",
    )
    size: int = Field(default=1024, ge=64, le=4096, description="Raster width in pixels (png, jpg, webp).")
    quiet_zone: int = Field(default=4, ge=0, le=10, description="White border around the code in modules (4 is the standard).")
    print_size_mm: Optional[float] = Field(default=None, ge=5, le=2000, description="Physical width for svg, pdf and eps in millimetres. Empty sizes them from `size`.")
    verify_scan: bool = Field(default=True, description="Decode the rendered code and report whether it scans back to the same content.")

    @model_validator(mode="after")
    def _check_content(self):
        ct = self.content_type
        need = {
            "url": ["data"], "text": ["data"], "phone": ["data"], "sms": ["data"],
            "whatsapp": ["data"], "email": ["data"], "wifi": ["wifi_ssid"],
            "geo": ["latitude", "longitude"], "event": ["event_title", "event_start"],
            "epc_payment": ["payment_name", "payment_iban", "payment_amount"],
        }.get(ct, [])
        missing = [f for f in need if getattr(self, f) in (None, "")]
        if ct in ("vcard", "mecard") and not (self.contact_first_name or self.contact_last_name):
            missing.append("contact_first_name or contact_last_name")
        if missing:
            raise ValueError("content_type '%s' needs: %s" % (ct, ", ".join(missing)))
        if ct == "wifi" and self.wifi_security != "nopass" and not self.wifi_password:
            raise ValueError("wifi_password is required unless wifi_security is 'nopass'")
        return self


class AppOutput(BaseAppOutput):
    image: File = Field(description="PNG preview of the QR code.")
    files: List[File] = Field(description="One file per requested output format.")
    encoded_data: str = Field(description="The exact text encoded in the QR code.")
    version: int = Field(description="QR version used (1-40).")
    error_correction: str = Field(description="Error correction level used (L, M, Q, H).")
    modules: int = Field(description="Grid size in modules, without the quiet zone.")
    scan_verified: Optional[bool] = Field(default=None, description="True if the rendered code decodes back to encoded_data; empty when verify_scan is off.")


# ---------------------------------------------------------------- payloads

def _clean(s):
    return (s or "").strip()


def _ical_dt(value: str):
    v = value.replace(" ", "T")
    if len(v) == 10:
        return ";VALUE=DATE:" + v.replace("-", "")
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    if re.search(r"[+-]\d{4}$", v):
        v = v[:-2] + ":" + v[-2:]
    dt = datetime.fromisoformat(v)
    if dt.tzinfo is not None:
        return ":" + dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ":" + dt.strftime("%Y%m%dT%H%M%S")


def _ical_text(s):
    return _clean(s).replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def build_payload(i: AppInput) -> str:
    ct = i.content_type
    d = _clean(i.data)
    if ct == "text":
        return i.data
    if ct == "url":
        return d if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", d) else "https://" + d
    if ct == "phone":
        return "tel:" + re.sub(r"[^0-9+*#]", "", d)
    if ct == "sms":
        num = re.sub(r"[^0-9+]", "", d)
        return "SMSTO:%s:%s" % (num, i.message or "")
    if ct == "whatsapp":
        num = re.sub(r"[^0-9]", "", d)
        url = "https://wa.me/" + num
        return url + ("?text=" + quote(i.message) if i.message else "")
    if ct == "email":
        return qrhelpers.make_make_email_data(to=d, subject=i.subject or None, body=i.message or None)
    if ct == "wifi":
        sec = None if i.wifi_security == "nopass" else i.wifi_security
        return qrhelpers.make_wifi_data(ssid=i.wifi_ssid, password=i.wifi_password if sec else None,
                                        security=sec or "nopass", hidden=i.wifi_hidden)
    if ct == "geo":
        return qrhelpers.make_geo_data(i.latitude, i.longitude)
    if ct == "vcard":
        first, last = _clean(i.contact_first_name), _clean(i.contact_last_name)
        return qrhelpers.make_vcard_data(
            name="%s;%s" % (last, first), displayname=(" ".join(p for p in (first, last) if p)),
            email=i.contact_email or None, phone=i.contact_phone or None, url=i.contact_website or None,
            org=i.contact_organization or None, title=i.contact_title or None, street=i.contact_street or None,
            city=i.contact_city or None, zipcode=i.contact_zip or None, country=i.contact_country or None,
            memo=i.contact_note or None,
        )
    if ct == "mecard":
        first, last = _clean(i.contact_first_name), _clean(i.contact_last_name)
        return qrhelpers.make_mecard_data(
            name=",".join(p for p in (last, first) if p), email=i.contact_email or None,
            phone=i.contact_phone or None, url=i.contact_website or None, memo=i.contact_note or None,
            city=i.contact_city or None, zipcode=i.contact_zip or None, country=i.contact_country or None,
        )
    if ct == "event":
        lines = ["BEGIN:VEVENT", "SUMMARY:" + _ical_text(i.event_title), "DTSTART" + _ical_dt(i.event_start)]
        if i.event_end:
            lines.append("DTEND" + _ical_dt(i.event_end))
        if i.event_location:
            lines.append("LOCATION:" + _ical_text(i.event_location))
        if i.event_description:
            lines.append("DESCRIPTION:" + _ical_text(i.event_description))
        lines.append("END:VEVENT")
        return "\n".join(lines)
    if ct == "epc_payment":
        lines = ["BCD", "002", "1", "SCT", _clean(i.payment_bic).upper(), _clean(i.payment_name),
                 re.sub(r"\s", "", i.payment_iban).upper(), "EUR%.2f" % i.payment_amount, "", "",
                 _clean(i.payment_reference)]
        return "\n".join(lines).rstrip("\n")
    raise ValueError("unknown content_type %s" % ct)


# ---------------------------------------------------------------- geometry

def f(v):
    s = ("%.3f" % v).rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def rrect(x, y, w, h, tl=0.0, tr=0.0, br=0.0, bl=0.0):
    """Rounded rectangle path, clockwise, per-corner radii."""
    p = ["M%s,%s" % (f(x + tl), f(y)), "H%s" % f(x + w - tr)]
    if tr:
        p.append("A%s,%s 0 0 1 %s,%s" % (f(tr), f(tr), f(x + w), f(y + tr)))
    p.append("V%s" % f(y + h - br))
    if br:
        p.append("A%s,%s 0 0 1 %s,%s" % (f(br), f(br), f(x + w - br), f(y + h)))
    p.append("H%s" % f(x + bl))
    if bl:
        p.append("A%s,%s 0 0 1 %s,%s" % (f(bl), f(bl), f(x), f(y + h - bl)))
    p.append("V%s" % f(y + tl))
    if tl:
        p.append("A%s,%s 0 0 1 %s,%s" % (f(tl), f(tl), f(x + tl), f(y)))
    p.append("Z")
    return "".join(p)


def circle(cx, cy, r):
    return "M%s,%sa%s,%s 0 1 0 %s,0a%s,%s 0 1 0 %s,0Z" % (
        f(cx - r), f(cy), f(r), f(r), f(2 * r), f(r), f(r), f(-2 * r))


def diamond(cx, cy, r):
    return "M%s,%sL%s,%sL%s,%sL%s,%sZ" % (f(cx), f(cy - r), f(cx + r), f(cy), f(cx), f(cy + r), f(cx - r), f(cy))


def shape(kind, x, y, s, sharp_corner):
    """A square-ish shape of side s at (x, y). sharp_corner: 'tl','tr','br','bl' for leaf."""
    if kind == "square":
        return rrect(x, y, s, s)
    if kind == "rounded":
        r = s * 0.25
        return rrect(x, y, s, s, r, r, r, r)
    if kind == "extra-rounded":
        r = s * 0.4
        return rrect(x, y, s, s, r, r, r, r)
    if kind == "circle":
        return circle(x + s / 2, y + s / 2, s / 2)
    if kind == "diamond":
        return diamond(x + s / 2, y + s / 2, s / 2)
    if kind == "leaf":
        r = s * 0.45
        radii = {"tl": r, "tr": r, "br": r, "bl": r}
        radii[sharp_corner] = 0
        return rrect(x, y, s, s, radii["tl"], radii["tr"], radii["br"], radii["bl"])
    raise ValueError(kind)


def module_paths(matrix, n, ox, oy, style, skip):
    dark = lambda r, c: 0 <= r < n and 0 <= c < n and matrix[r][c] and not skip(r, c)
    out = []
    if style in ("horizontal-bars", "vertical-bars"):
        horiz = style == "horizontal-bars"
        g = 0.1
        for a in range(n):
            b = 0
            while b < n:
                r, c = (a, b) if horiz else (b, a)
                if not dark(r, c):
                    b += 1
                    continue
                start = b
                while b < n and dark(*((a, b) if horiz else (b, a))):
                    b += 1
                length = b - start
                if horiz:
                    x, y, w, h = ox + start + g, oy + a + g, length - 2 * g, 1 - 2 * g
                else:
                    x, y, w, h = ox + a + g, oy + start + g, 1 - 2 * g, length - 2 * g
                rad = (1 - 2 * g) / 2
                out.append(rrect(x, y, w, h, rad, rad, rad, rad))
        return out
    for r in range(n):
        for c in range(n):
            if not dark(r, c):
                continue
            x, y = ox + c, oy + r
            if style == "square":
                out.append(rrect(x, y, 1, 1))
            elif style == "rounded":
                out.append(rrect(x + 0.05, y + 0.05, 0.9, 0.9, 0.3, 0.3, 0.3, 0.3))
            elif style == "dots":
                out.append(circle(x + 0.5, y + 0.5, 0.45))
            elif style == "diamond":
                out.append(diamond(x + 0.5, y + 0.5, 0.55))
            elif style == "small-squares":
                out.append(rrect(x + 0.15, y + 0.15, 0.7, 0.7))
            elif style == "leaf":
                out.append(rrect(x + 0.05, y + 0.05, 0.9, 0.9, 0.45, 0, 0.45, 0))
            elif style == "fluid":
                up, dn, lf, rt = dark(r - 1, c), dark(r + 1, c), dark(r, c - 1), dark(r, c + 1)
                k = 0.5
                out.append(rrect(x, y, 1, 1,
                                 0 if (up or lf) else k, 0 if (up or rt) else k,
                                 0 if (dn or rt) else k, 0 if (dn or lf) else k))
    return out


# ---------------------------------------------------------------- colors

def parse_color(c):
    """-> (#rrggbb, opacity) or (None, 0) for transparent."""
    if c == "transparent":
        return None, 0.0
    h = c[1:]
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    alpha = int(h[6:8], 16) / 255 if len(h) == 8 else 1.0
    return "#" + h[:6].lower(), alpha


def fill_attr(c):
    rgb, a = parse_color(c)
    if rgb is None:
        return 'fill="none"'
    return 'fill="%s"%s' % (rgb, (' fill-opacity="%s"' % f(a)) if a < 1 else "")


def luminance(c):
    rgb, _ = parse_color(c)
    if rgb is None:
        return 1.0
    r, g, b = (int(rgb[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# ---------------------------------------------------------------- logo

def logo_data_uri(file: File):
    path = file.path
    name = (file.filename or "") + " " + (path or "") + " " + (file.uri or "")
    ctype = (file.content_type or "").lower()
    if not ctype or ctype == "application/octet-stream":
        ctype = (mimetypes.guess_type(path)[0] or mimetypes.guess_type(name.split()[0] if name.strip() else "")[0] or "")
    with open(path, "rb") as fh:
        raw = fh.read()
    head = raw[:512].lstrip().lower()
    if "svg" in ctype or name.lower().rstrip().endswith(".svg") or head.startswith(b"<svg") or (head.startswith(b"<?xml") and b"<svg" in raw[:4096].lower()):
        return "data:image/svg+xml;base64," + base64.b64encode(raw).decode(), 1.0
    try:
        img = Image.open(io.BytesIO(raw))
        img.seek(0)
        img = img.convert("RGBA")
    except Exception as e:
        raise ValueError("logo is not a readable image (PNG, JPG, WEBP, GIF or SVG): %s" % e)
    img.thumbnail((1024, 1024))  # downsizes only, never enlarges
    aspect = img.height / img.width if img.width else 1.0
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), aspect


# ---------------------------------------------------------------- svg build

def build_svg(i: AppInput, qr, logo):
    matrix = qr.matrix
    n = len(matrix)
    q = i.quiet_zone
    W = n + 2 * q
    has_label = bool(_clean(i.label_text))
    framed = i.frame_style != "none"
    band = W * 0.17 if has_label else 0.0
    pad = max(1.0, W * 0.04) if framed else 0.0
    cw, ch = W + 2 * pad, W + band + 2 * pad
    code_x = pad
    code_y = pad + (band if (has_label and i.label_position == "top") else 0.0)
    ox, oy = code_x + q, code_y + q  # module (0,0)
    corner = W * 0.06 if framed else 0.0

    parts, defs = [], []

    # background
    if i.background_color != "transparent":
        parts.append('<path d="%s" %s/>' % (rrect(0, 0, cw, ch, corner, corner, corner, corner), fill_attr(i.background_color)))

    # gradient
    code_fill = fill_attr(i.foreground_color)
    if i.gradient_type != "none":
        s_rgb, s_a = parse_color(i.gradient_start_color)
        e_rgb, e_a = parse_color(i.gradient_end_color)
        stops = ('<stop offset="0" stop-color="%s" stop-opacity="%s"/><stop offset="1" stop-color="%s" stop-opacity="%s"/>'
                 % (s_rgb, f(s_a), e_rgb, f(e_a)))
        cx, cy = ox + n / 2, oy + n / 2
        if i.gradient_type == "linear":
            a = math.radians(i.gradient_angle)
            dx, dy = math.cos(a) * n / 2, math.sin(a) * n / 2
            defs.append('<linearGradient id="g" gradientUnits="userSpaceOnUse" x1="%s" y1="%s" x2="%s" y2="%s">%s</linearGradient>'
                        % (f(cx - dx), f(cy - dy), f(cx + dx), f(cy + dy), stops))
        else:
            defs.append('<radialGradient id="g" gradientUnits="userSpaceOnUse" cx="%s" cy="%s" r="%s">%s</radialGradient>'
                        % (f(cx), f(cy), f(n * 0.71), stops))
        code_fill = 'fill="url(#g)"'

    # finder patterns
    finders = [(0, 0, "br"), (0, n - 7, "bl"), (n - 7, 0, "tr")]

    def in_finder(r, c):
        return (r < 7 and c < 7) or (r < 7 and c >= n - 7) or (r >= n - 7 and c < 7)

    # logo area
    logo_box = None
    if logo is not None:
        lw = n * i.logo_size
        lh = lw * logo[1]
        if lh > n * 0.3:
            lh = n * 0.3
            lw = lh / logo[1]
        m = i.logo_margin
        logo_box = (ox + (n - lw) / 2, oy + (n - lh) / 2, lw, lh)
        clear = (logo_box[0] - m, logo_box[1] - m, lw + 2 * m, lh + 2 * m)

        def in_logo(r, c):
            x0, y0 = ox + c, oy + r
            if i.logo_shape == "circle":
                ccx, ccy = clear[0] + clear[2] / 2, clear[1] + clear[3] / 2
                rad = max(clear[2], clear[3]) / 2
                nx, ny = min(max(ccx, x0), x0 + 1), min(max(ccy, y0), y0 + 1)
                return (nx - ccx) ** 2 + (ny - ccy) ** 2 < rad ** 2
            return x0 + 1 > clear[0] and x0 < clear[0] + clear[2] and y0 + 1 > clear[1] and y0 < clear[1] + clear[3]
    else:
        in_logo = lambda r, c: False

    mods = module_paths(matrix, n, ox, oy, i.module_style, lambda r, c: in_finder(r, c) or in_logo(r, c))
    if mods:
        parts.append('<path d="%s" %s/>' % ("".join(mods), code_fill))

    frame_fill = fill_attr(i.eye_frame_color) if i.eye_frame_color else code_fill
    ball_fill = fill_attr(i.eye_ball_color) if i.eye_ball_color else code_fill
    rings, balls = [], []
    for fr, fc, sharp in finders:
        x, y = ox + fc, oy + fr
        if i.eye_frame_style == "circle":
            rings.append(circle(x + 3.5, y + 3.5, 3.5) + circle(x + 3.5, y + 3.5, 2.5))
        else:
            outer = shape(i.eye_frame_style, x, y, 7, sharp)
            inner_kind = "square" if i.eye_frame_style == "square" else i.eye_frame_style
            # inner hole: keep the same corner style, scaled to 5x5
            if inner_kind == "leaf":
                inner = shape("leaf", x + 1, y + 1, 5, sharp)
            elif inner_kind in ("rounded", "extra-rounded"):
                r = (7 * (0.25 if inner_kind == "rounded" else 0.4)) - 1
                r = max(r, 0.0)
                inner = rrect(x + 1, y + 1, 5, 5, r, r, r, r)
            else:
                inner = rrect(x + 1, y + 1, 5, 5)
            rings.append(outer + inner)
        balls.append(shape(i.eye_ball_style, x + 2, y + 2, 3, sharp))
    parts.append('<path d="%s" fill-rule="evenodd" %s/>' % ("".join(rings), frame_fill))
    parts.append('<path d="%s" %s/>' % ("".join(balls), ball_fill))

    # logo
    if logo_box is not None:
        if i.logo_background_color:
            cx_, cy_, cw_, ch_ = clear
            if i.logo_shape == "circle":
                d = circle(cx_ + cw_ / 2, cy_ + ch_ / 2, max(cw_, ch_) / 2)
            elif i.logo_shape == "rounded":
                r = min(cw_, ch_) * 0.2
                d = rrect(cx_, cy_, cw_, ch_, r, r, r, r)
            else:
                d = rrect(cx_, cy_, cw_, ch_)
            parts.append('<path d="%s" %s/>' % (d, fill_attr(i.logo_background_color)))
        parts.append('<image x="%s" y="%s" width="%s" height="%s" preserveAspectRatio="xMidYMid meet" xlink:href="%s" href="%s"/>'
                     % (f(logo_box[0]), f(logo_box[1]), f(logo_box[2]), f(logo_box[3]), logo[0], logo[0]))

    # frame
    if framed:
        sw = W * 0.025
        if i.frame_style == "banner" and has_label:
            by = 0 if i.label_position == "top" else pad + W
            bh = band + pad
            if i.label_position == "top":
                d = rrect(0, 0, cw, bh, corner, corner, 0, 0)
            else:
                d = rrect(0, by, cw, bh, 0, 0, corner, corner)
            parts.append('<path d="%s" %s/>' % (d, fill_attr(i.frame_color)))
        r = max(corner - sw / 2, 0)
        parts.append('<path d="%s" fill="none" stroke="%s" stroke-width="%s"%s/>' % (
            rrect(sw / 2, sw / 2, cw - sw, ch - sw, r, r, r, r), parse_color(i.frame_color)[0], f(sw),
            (' stroke-opacity="%s"' % f(parse_color(i.frame_color)[1])) if parse_color(i.frame_color)[1] < 1 else ""))

    # label
    if has_label:
        text = _clean(i.label_text)
        if i.label_color:
            lc = i.label_color
        elif i.frame_style == "banner":
            lc = i.background_color if i.background_color != "transparent" else "#FFFFFF"
        else:
            lc = i.frame_color if framed else i.foreground_color
        fs = min(band * 0.55, (cw * 0.88) / max(1, 0.62 * len(text)))
        if i.label_position == "top":
            mid = (pad + band) / 2 + (pad / 2 if i.frame_style != "banner" else 0)
        else:
            mid = pad + W + (band + (pad if i.frame_style == "banner" else 0)) / 2
        parts.append('<text x="%s" y="%s" text-anchor="middle" font-family="DejaVu Sans, Arial, Helvetica, sans-serif" font-weight="bold" font-size="%s" %s>%s</text>'
                     % (f(cw / 2), f(mid + fs * 0.36), f(fs), fill_attr(lc), escape(text)))

    if i.print_size_mm:
        wattr = "%smm" % f(i.print_size_mm)
        hattr = "%smm" % f(i.print_size_mm * ch / cw)
    else:
        wattr = str(i.size)
        hattr = str(int(round(i.size * ch / cw)))
    svg = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
           'width="%s" height="%s" viewBox="0 0 %s %s" shape-rendering="geometricPrecision">'
           '%s%s</svg>\n') % (wattr, hattr, f(cw), f(ch), ("<defs>%s</defs>" % "".join(defs)) if defs else "", "".join(parts))
    return svg, cw, ch


# ---------------------------------------------------------------- app

class App(BaseApp):
    async def setup(self, config=None):
        self.logger = logging.getLogger(__name__)

    async def run(self, input_data: AppInput) -> AppOutput:
        i = input_data
        payload = build_payload(i)
        self.logger.info("content_type=%s payload_chars=%d", i.content_type, len(payload))

        logo = logo_data_uri(i.logo) if i.logo is not None else None

        if i.content_type == "epc_payment":
            error = "M"
        elif i.error_correction == "auto":
            error = "H" if logo else "M"
        else:
            error = i.error_correction
        try:
            qr = segno.make(payload, error=error.lower(), version=i.qr_version, micro=False,
                            boost_error=(i.error_correction == "auto" and i.content_type != "epc_payment"))
        except segno.DataOverflowError as e:
            raise ValueError("content does not fit: %s. Shorten it, lower error_correction or raise/clear qr_version." % e)
        n = len(qr.matrix)
        self.logger.info("qr version=%s error=%s modules=%d", qr.version, qr.error, n)
        if luminance(i.foreground_color) > luminance(i.background_color) and i.background_color != "transparent":
            self.logger.warning("foreground is lighter than background (inverted); some scanners cannot read inverted codes")

        svg, cw, ch = build_svg(i, qr, logo)
        svg_bytes = svg.encode("utf-8")
        out_w = i.size
        out_h = int(round(i.size * ch / cw))

        outdir = tempfile.mkdtemp(prefix="qr-")
        png_path = os.path.join(outdir, "qr-code.png")
        cairosvg.svg2png(bytestring=svg_bytes, write_to=png_path, output_width=out_w, output_height=out_h)

        files = []
        formats = list(dict.fromkeys(i.output_formats))
        for fmt in formats:
            path = os.path.join(outdir, "qr-code.%s" % fmt)
            if fmt == "png":
                pass
            elif fmt == "svg":
                with open(path, "wb") as fh:
                    fh.write(svg_bytes)
            elif fmt == "pdf":
                cairosvg.svg2pdf(bytestring=svg_bytes, write_to=path)
            elif fmt == "eps":
                cairosvg.svg2eps(bytestring=svg_bytes, write_to=path)
            elif fmt in ("jpg", "webp"):
                img = Image.open(png_path)
                if fmt == "jpg":
                    flat = Image.new("RGB", img.size, (255, 255, 255))
                    flat.paste(img, mask=img.convert("RGBA").split()[3])
                    flat.save(path, format="JPEG", quality=95, subsampling=0)
                else:
                    img.save(path, format="WEBP", lossless=True)
            files.append(File(path=path))
            self.logger.info("wrote %s (%d bytes)", os.path.basename(path), os.path.getsize(path))

        scan_ok = None
        if i.verify_scan:
            scan_ok = self._verify(png_path, payload, i, svg_bytes, cw, ch)

        return AppOutput(
            image=File(path=png_path),
            files=files,
            encoded_data=payload,
            version=int(qr.version),
            error_correction=str(qr.error).upper(),
            modules=n,
            scan_verified=scan_ok,
            output_meta=OutputMeta(outputs=[ImageMeta(width=out_w, height=out_h, count=1)]),
        )

    def _verify(self, png_path, payload, i, svg_bytes, cw, ch):
        try:
            import zxingcpp
        except Exception as e:  # pragma: no cover
            self.logger.warning("scan check unavailable: %s", e)
            return None
        # decode at a comfortable size, flattened on white (or black for light-on-transparent)
        w = max(600, min(i.size, 1600))
        buf = io.BytesIO(cairosvg.svg2png(bytestring=svg_bytes, output_width=w, output_height=int(round(w * ch / cw))))
        img = Image.open(buf).convert("RGBA")
        bg = (255, 255, 255)
        if i.background_color == "transparent" and luminance(i.foreground_color) > 0.6:
            bg = (0, 0, 0)
        flat = Image.new("RGB", img.size, bg)
        flat.paste(img, mask=img.split()[3])
        for candidate in (flat, Image.eval(flat, lambda p: 255 - p)):
            try:
                results = zxingcpp.read_barcodes(candidate)
            except Exception as e:
                self.logger.warning("scan check failed to run: %s", e)
                return None
            for r in results:
                if r.text == payload:
                    self.logger.info("scan check passed")
                    return True
        self.logger.warning("scan check FAILED: the styled code did not decode back to its content; "
                            "try error_correction H, a smaller logo, higher contrast or square modules")
        return False

    async def unload(self):
        pass
