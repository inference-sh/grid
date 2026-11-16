"""Vector QR rendering: one SVG master, drawn from the QR matrix and the style options."""
import base64
import io
import math
import mimetypes
from xml.sax.saxutils import escape

from PIL import Image


def _clean(s):
    return (s or "").strip()


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

def logo_data_uri(file):
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

def build_svg(i, qr, logo):
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
