import io
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import List, Literal, Optional
from urllib.parse import quote

import cairosvg
import segno
import segno.helpers as qrhelpers
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, ImageMeta, OutputMeta, function

from .render import build_svg, logo_data_uri, luminance

HEX = r"#([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})"
COLOR = r"^%s$" % HEX
COLOR_OR_TRANSPARENT = r"^(%s|transparent)$" % HEX
COLOR_OR_AUTO = r"^(%s|auto)$" % HEX
ISO_DATETIME = r"\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?(Z|[+-]\d{2}:?\d{2})?"

ModuleShape = Literal[
    "square", "rounded", "dots", "diamond", "fluid", "leaf",
    "small-squares", "horizontal-bars", "vertical-bars",
]
EyeFrameShape = Literal["square", "rounded", "extra-rounded", "circle", "leaf"]
EyeBallShape = Literal["square", "rounded", "circle", "diamond", "leaf"]
FORMATS = ("png", "svg", "pdf", "eps", "jpg", "webp")


# ================================================================ option groups
# Every group and field has a concrete default (no null), so forms and callers can always send a full object.
# "auto" / "" / 0 mean "use the automatic value".

class StyleOptions(BaseModel):
    """Shapes and colors of the code."""
    module_shape: ModuleShape = Field(default="square", description="Shape of the data dots.")
    eye_frame_shape: EyeFrameShape = Field(default="square", description="Shape of the outer ring of the three corner eyes.")
    eye_ball_shape: EyeBallShape = Field(default="square", description="Shape of the center of the three corner eyes.")
    color: str = Field(default="#000000", pattern=COLOR, description="Code color, hex #RGB, #RRGGBB or #RRGGBBAA.")
    background_color: str = Field(default="#FFFFFF", pattern=COLOR_OR_TRANSPARENT, description="Background color, hex, or 'transparent' (JPG falls back to white).")
    eye_frame_color: str = Field(default="auto", pattern=COLOR_OR_AUTO, description="Eye ring color, hex, or 'auto' for the code color or gradient.")
    eye_ball_color: str = Field(default="auto", pattern=COLOR_OR_AUTO, description="Eye center color, hex, or 'auto' for the code color or gradient.")
    gradient: Literal["none", "linear", "radial"] = Field(default="none", description="Fill the code with a gradient instead of the code color.")
    gradient_start_color: str = Field(default="#000000", pattern=COLOR, description="Gradient start color (center for radial).")
    gradient_end_color: str = Field(default="#3B5BDB", pattern=COLOR, description="Gradient end color (edge for radial).")
    gradient_angle: int = Field(default=45, ge=0, le=360, description="Linear gradient angle in degrees (0 left to right, 90 top to bottom).")


class LogoOptions(BaseModel):
    """A logo in the center of the code. Leave image empty for no logo."""
    image: Optional[File] = Field(default=None, description="Logo file: PNG, JPG, WEBP, GIF or SVG. Empty for no logo.")
    size: float = Field(default=0.22, ge=0.05, le=0.3, description="Logo width as a fraction of the code width (max 0.3 keeps it scannable).")
    margin: float = Field(default=1.0, ge=0, le=4, description="Empty space around the logo, in modules.")
    shape: Literal["square", "rounded", "circle"] = Field(default="rounded", description="Shape of the area cleared behind the logo.")
    background_color: str = Field(default="transparent", pattern=COLOR_OR_TRANSPARENT, description="Fill the cleared area with this color, or 'transparent' to show the background.")


class FrameOptions(BaseModel):
    """A frame and caption around the code. Off by default."""
    style: Literal["none", "border", "banner"] = Field(default="none", description="none, a rounded border, or a border with a filled caption banner (like 'SCAN ME' frames).")
    color: str = Field(default="#000000", pattern=COLOR, description="Border and banner color.")
    text: str = Field(default="", max_length=60, description="Caption, e.g. 'SCAN ME'. Empty for none. Works with every style.")
    position: Literal["bottom", "top"] = Field(default="bottom", description="Where the caption goes.")
    text_color: str = Field(default="auto", pattern=COLOR_OR_AUTO, description="Caption color, or 'auto': background color on a banner, frame color otherwise.")


class OutputOptions(BaseModel):
    """File formats, size and encoding. The PNG preview in `image` is always returned."""
    png: bool = Field(default=True, description="PNG file (raster, `size` pixels wide, keeps transparency).")
    svg: bool = Field(default=True, description="SVG file (vector, scales to any size).")
    pdf: bool = Field(default=False, description="PDF file (vector, for print).")
    eps: bool = Field(default=False, description="EPS file (vector, for print and design tools).")
    jpg: bool = Field(default=False, description="JPG file (raster, white background).")
    webp: bool = Field(default=False, description="WEBP file (raster, lossless, keeps transparency).")
    size: int = Field(default=1024, ge=64, le=4096, description="Raster width in pixels (png, jpg, webp).")
    print_size_mm: float = Field(default=0, ge=0, le=2000, description="Physical width of svg, pdf and eps in millimetres. 0 sizes them from `size`.")
    quiet_zone: int = Field(default=4, ge=0, le=10, description="Blank border around the code in modules (4 is the standard).")
    error_correction: Literal["auto", "L", "M", "Q", "H"] = Field(default="auto", description="L 7%, M 15%, Q 25%, H 30%. auto picks M, or H with a logo.")
    qr_version: int = Field(default=0, ge=0, le=40, description="Force a QR version (1-40). 0 picks the smallest that fits.")
    verify_scan: bool = Field(default=True, description="Decode the rendered code and report whether it scans back to its content.")


class QROptions(BaseAppInput):
    style: StyleOptions = Field(default_factory=StyleOptions, description="Shapes, colors and gradient.")
    logo: LogoOptions = Field(default_factory=LogoOptions, description="Logo in the center (off until an image is set).")
    frame: FrameOptions = Field(default_factory=FrameOptions, description="Frame and caption (off by default).")
    output: OutputOptions = Field(default_factory=OutputOptions, description="Formats, size and encoding.")


# ================================================================ function inputs
# Content fields come first (QROptions is listed first as a base so its fields are collected last).
# Optional text fields default to "" rather than null.

class UrlContent(BaseAppInput):
    url: str = Field(min_length=1, max_length=2000, description="Link to open. https:// is added when no scheme is given.")


class TextContent(BaseAppInput):
    text: str = Field(min_length=1, max_length=4000, description="Any text.")


class WifiContent(BaseAppInput):
    ssid: str = Field(min_length=1, max_length=64, description="Network name.")
    password: str = Field(default="", max_length=128, description="Network password. Required unless security is nopass.")
    security: Literal["WPA", "WEP", "nopass"] = Field(default="WPA", description="WPA covers WPA2 and WPA3.")
    hidden: bool = Field(default=False, description="The network is hidden.")


class ContactContent(BaseAppInput):
    first_name: str = Field(min_length=1, max_length=100, description="First name.")
    last_name: str = Field(default="", max_length=100, description="Last name.")
    organization: str = Field(default="", max_length=200, description="Company (vCard only).")
    title: str = Field(default="", max_length=200, description="Job title (vCard only).")
    phone: str = Field(default="", max_length=40, description="Phone number.")
    email: str = Field(default="", max_length=200, description="Email address.")
    website: str = Field(default="", max_length=300, description="Website.")
    street: str = Field(default="", max_length=200, description="Street address (vCard only).")
    city: str = Field(default="", max_length=100, description="City.")
    zip: str = Field(default="", max_length=20, description="Postal code.")
    country: str = Field(default="", max_length=100, description="Country.")
    note: str = Field(default="", max_length=500, description="Note.")
    format: Literal["vcard", "mecard"] = Field(default="vcard", description="vCard (most fields, every phone) or MeCard (shorter, smaller code).")


class EmailContent(BaseAppInput):
    to: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=254, description="Recipient address.")
    subject: str = Field(default="", max_length=300, description="Subject.")
    body: str = Field(default="", max_length=2000, description="Message body.")


class PhoneContent(BaseAppInput):
    number: str = Field(pattern=r"^\+?[0-9 ()./-]{3,30}$", description="Phone number, ideally with country code (+1 555 123 4567).")
    action: Literal["call", "sms", "whatsapp"] = Field(default="call", description="Call the number, start an SMS, or open a WhatsApp chat.")
    message: str = Field(default="", max_length=1000, description="Prefilled message (sms, whatsapp).")


class LocationContent(BaseAppInput):
    latitude: float = Field(ge=-90, le=90, description="Latitude.")
    longitude: float = Field(ge=-180, le=180, description="Longitude.")


class EventContent(BaseAppInput):
    title: str = Field(min_length=1, max_length=200, description="Event title.")
    start: str = Field(pattern=r"^%s$" % ISO_DATETIME, description="Start, ISO 8601: 2025-06-01 (all day), 2025-06-01T18:00, or with Z/offset.")
    end: str = Field(default="", pattern=r"^(%s)?$" % ISO_DATETIME, description="End, ISO 8601. Empty for none.")
    location: str = Field(default="", max_length=300, description="Location.")
    description: str = Field(default="", max_length=1000, description="Description.")


class PaymentContent(BaseAppInput):
    name: str = Field(min_length=1, max_length=70, description="Beneficiary name.")
    iban: str = Field(pattern=r"^[A-Za-z]{2}[0-9]{2}[A-Za-z0-9 ]{10,36}$", description="Beneficiary IBAN.")
    amount: float = Field(ge=0.01, le=999999999.99, description="Amount in EUR.")
    bic: str = Field(default="", pattern=r"^([A-Za-z0-9]{8}([A-Za-z0-9]{3})?)?$", description="Beneficiary BIC (optional inside the EEA).")
    reference: str = Field(default="", max_length=140, description="Remittance text shown to the payer.")


class UrlInput(QROptions, UrlContent):
    pass


class TextInput(QROptions, TextContent):
    pass


class WifiInput(QROptions, WifiContent):
    @model_validator(mode="after")
    def _password(self):
        if self.security != "nopass" and not self.password:
            raise ValueError("password is required unless security is 'nopass'")
        return self


class ContactInput(QROptions, ContactContent):
    pass


class EmailInput(QROptions, EmailContent):
    pass


class PhoneInput(QROptions, PhoneContent):
    pass


class LocationInput(QROptions, LocationContent):
    pass


class EventInput(QROptions, EventContent):
    pass


class PaymentInput(QROptions, PaymentContent):
    pass


class QROutput(BaseAppOutput):
    image: File = Field(description="PNG preview of the QR code.")
    files: List[File] = Field(description="One file per format switched on in `output`.")
    encoded_data: str = Field(description="The exact text encoded in the QR code.")
    version: int = Field(description="QR version used (1-40).")
    error_correction: str = Field(description="Error correction level used (L, M, Q, H).")
    modules: int = Field(description="Grid size in modules, without the quiet zone.")
    scan_verified: Optional[bool] = Field(default=None, description="True if the rendered code decodes back to encoded_data; empty when verify_scan is off.")


# ================================================================ payloads

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


def url_payload(i: UrlInput):
    u = i.url.strip()
    return u if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", u) else "https://" + u


def wifi_payload(i: WifiInput):
    if i.security == "nopass":
        return qrhelpers.make_wifi_data(ssid=i.ssid, password=None, security="nopass", hidden=i.hidden)
    return qrhelpers.make_wifi_data(ssid=i.ssid, password=i.password, security=i.security, hidden=i.hidden)


def contact_payload(i: ContactInput):
    first, last = _clean(i.first_name), _clean(i.last_name)
    if i.format == "mecard":
        return qrhelpers.make_mecard_data(
            name=",".join(p for p in (last, first) if p), email=i.email or None, phone=i.phone or None,
            url=i.website or None, memo=i.note or None, city=i.city or None, zipcode=i.zip or None,
            country=i.country or None,
        )
    return qrhelpers.make_vcard_data(
        name="%s;%s" % (last, first), displayname=" ".join(p for p in (first, last) if p),
        email=i.email or None, phone=i.phone or None, url=i.website or None, org=i.organization or None,
        title=i.title or None, street=i.street or None, city=i.city or None, zipcode=i.zip or None,
        country=i.country or None, memo=i.note or None,
    )


def phone_payload(i: PhoneInput):
    if i.action == "call":
        return "tel:" + re.sub(r"[^0-9+]", "", i.number)
    if i.action == "sms":
        return "SMSTO:%s:%s" % (re.sub(r"[^0-9+]", "", i.number), i.message or "")
    url = "https://wa.me/" + re.sub(r"[^0-9]", "", i.number)
    return url + ("?text=" + quote(i.message) if i.message else "")


def event_payload(i: EventInput):
    lines = ["BEGIN:VEVENT", "SUMMARY:" + _ical_text(i.title), "DTSTART" + _ical_dt(i.start)]
    if i.end:
        lines.append("DTEND" + _ical_dt(i.end))
    if i.location:
        lines.append("LOCATION:" + _ical_text(i.location))
    if i.description:
        lines.append("DESCRIPTION:" + _ical_text(i.description))
    lines.append("END:VEVENT")
    return "\n".join(lines)


def payment_payload(i: PaymentInput):
    # EPC069-12 v002, UTF-8, SEPA credit transfer
    lines = ["BCD", "002", "1", "SCT", _clean(i.bic).upper(), _clean(i.name),
             re.sub(r"\s", "", i.iban).upper(), "EUR%.2f" % i.amount, "", "", _clean(i.reference)]
    return "\n".join(lines).rstrip("\n")


def _unset(value, *sentinels):
    """Map the 'automatic' defaults ("auto", "transparent", "", 0) to None for the renderer."""
    return None if value in sentinels else value


def _flat(o: QROptions):
    """Map the option groups onto the flat names the renderer reads."""
    s, lg, fr, out = o.style, o.logo, o.frame, o.output
    return SimpleNamespace(
        module_style=s.module_shape, eye_frame_style=s.eye_frame_shape, eye_ball_style=s.eye_ball_shape,
        foreground_color=s.color, background_color=s.background_color,
        eye_frame_color=_unset(s.eye_frame_color, "auto"), eye_ball_color=_unset(s.eye_ball_color, "auto"),
        gradient_type=s.gradient, gradient_start_color=s.gradient_start_color,
        gradient_end_color=s.gradient_end_color, gradient_angle=s.gradient_angle,
        logo_size=lg.size, logo_margin=lg.margin, logo_shape=lg.shape,
        logo_background_color=_unset(lg.background_color, "transparent"),
        frame_style=fr.style, frame_color=fr.color, label_text=_unset(fr.text.strip(), ""),
        label_position=fr.position, label_color=_unset(fr.text_color, "auto"),
        size=out.size, quiet_zone=out.quiet_zone, print_size_mm=_unset(out.print_size_mm, 0),
    )


# ================================================================ app

RO = {"readOnlyHint": True, "openWorldHint": False, "idempotentHint": True}


class App(BaseApp):
    async def setup(self, config=None):
        self.logger = logging.getLogger(__name__)

    @function(annotations={"title": "Link QR code", **RO})
    async def url(self, input_data: UrlInput) -> QROutput:
        """QR code that opens a link."""
        return self._make(url_payload(input_data), input_data)

    @function(annotations={"title": "Text QR code", **RO})
    async def text(self, input_data: TextInput) -> QROutput:
        """QR code holding plain text."""
        return self._make(input_data.text, input_data)

    @function(annotations={"title": "Wi-Fi QR code", **RO})
    async def wifi(self, input_data: WifiInput) -> QROutput:
        """QR code that joins a Wi-Fi network."""
        return self._make(wifi_payload(input_data), input_data)

    @function(annotations={"title": "Contact QR code", **RO})
    async def contact(self, input_data: ContactInput) -> QROutput:
        """QR code that saves a contact card (vCard or MeCard)."""
        return self._make(contact_payload(input_data), input_data)

    @function(annotations={"title": "Email QR code", **RO})
    async def email(self, input_data: EmailInput) -> QROutput:
        """QR code that starts an email with recipient, subject and body."""
        payload = qrhelpers.make_make_email_data(to=input_data.to.strip(), subject=input_data.subject or None,
                                                 body=input_data.body or None)
        return self._make(payload, input_data)

    @function(annotations={"title": "Phone, SMS or WhatsApp QR code", **RO})
    async def phone(self, input_data: PhoneInput) -> QROutput:
        """QR code that calls a number, starts an SMS or opens a WhatsApp chat."""
        return self._make(phone_payload(input_data), input_data)

    @function(annotations={"title": "Location QR code", **RO})
    async def location(self, input_data: LocationInput) -> QROutput:
        """QR code that opens a map location."""
        return self._make(qrhelpers.make_geo_data(input_data.latitude, input_data.longitude), input_data)

    @function(annotations={"title": "Calendar event QR code", **RO})
    async def event(self, input_data: EventInput) -> QROutput:
        """QR code that adds a calendar event."""
        return self._make(event_payload(input_data), input_data)

    @function(annotations={"title": "SEPA payment QR code", **RO})
    async def payment(self, input_data: PaymentInput) -> QROutput:
        """EPC (GiroCode) QR code for a SEPA euro bank transfer."""
        return self._make(payment_payload(input_data), input_data, force_error="M")

    # ------------------------------------------------------------ rendering

    def _make(self, payload: str, o: QROptions, force_error: Optional[str] = None) -> QROutput:
        out = o.output
        formats = [fmt for fmt in FORMATS if getattr(out, fmt)]
        self.logger.info("payload_chars=%d logo=%s frame=%s formats=%s", len(payload), o.logo.image is not None,
                         o.frame.style, formats)
        logo = logo_data_uri(o.logo.image) if o.logo.image is not None else None

        if force_error:
            error, boost = force_error, False
        elif out.error_correction == "auto":
            error, boost = ("H" if logo else "M"), True
        else:
            error, boost = out.error_correction, False
        try:
            qr = segno.make(payload, error=error.lower(), version=out.qr_version or None, micro=False, boost_error=boost)
        except segno.DataOverflowError as e:
            raise ValueError("content does not fit: %s. Shorten it, lower error_correction or set qr_version to 0." % e)
        n = len(qr.matrix)
        self.logger.info("qr version=%s error=%s modules=%d", qr.version, qr.error, n)

        r = _flat(o)
        if r.background_color != "transparent" and luminance(r.foreground_color) > luminance(r.background_color):
            self.logger.warning("code is lighter than background (inverted); some scanners cannot read inverted codes")

        svg, cw, ch = build_svg(r, qr, logo)
        svg_bytes = svg.encode("utf-8")
        out_w, out_h = out.size, int(round(out.size * ch / cw))

        outdir = tempfile.mkdtemp(prefix="qr-")
        png_path = os.path.join(outdir, "qr-code.png")
        cairosvg.svg2png(bytestring=svg_bytes, write_to=png_path, output_width=out_w, output_height=out_h)

        files = []
        for fmt in formats:
            path = os.path.join(outdir, "qr-code.%s" % fmt)
            if fmt == "svg":
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

        scan_ok = self._verify(payload, r, svg_bytes, cw, ch) if out.verify_scan else None
        return QROutput(
            image=File(path=png_path),
            files=files,
            encoded_data=payload,
            version=int(qr.version),
            error_correction=str(qr.error).upper(),
            modules=n,
            scan_verified=scan_ok,
            output_meta=OutputMeta(outputs=[ImageMeta(width=out_w, height=out_h, count=1)]),
        )

    def _verify(self, payload, r, svg_bytes, cw, ch):
        try:
            import zxingcpp
        except Exception as e:
            self.logger.warning("scan check unavailable: %s", e)
            return None
        w = max(600, min(r.size, 1600))
        img = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg_bytes, output_width=w,
                                                     output_height=int(round(w * ch / cw))))).convert("RGBA")
        bg = (0, 0, 0) if (r.background_color == "transparent" and luminance(r.foreground_color) > 0.6) else (255, 255, 255)
        flat = Image.new("RGB", img.size, bg)
        flat.paste(img, mask=img.split()[3])
        for candidate in (flat, Image.eval(flat, lambda p: 255 - p)):
            try:
                results = zxingcpp.read_barcodes(candidate)
            except Exception as e:
                self.logger.warning("scan check failed to run: %s", e)
                return None
            if any(x.text == payload for x in results):
                self.logger.info("scan check passed")
                return True
        self.logger.warning("scan check FAILED: the styled code did not decode back to its content; "
                            "try error_correction H, a smaller logo, higher contrast or square modules")
        return False

    async def unload(self):
        pass
