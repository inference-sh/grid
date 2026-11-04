from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

from retry_after import retry_delay


def test_seconds():
    assert retry_delay({"Retry-After": "7"}, 2) == 7


def test_absent_or_garbage_uses_default():
    assert retry_delay({}, 2) == 2
    assert retry_delay({"Retry-After": "soon"}, 2) == 2
    assert retry_delay({"Retry-After": "inf"}, 2) == 2


def test_http_date():
    when = datetime.now(timezone.utc) + timedelta(seconds=30)
    assert 25 < retry_delay({"Retry-After": format_datetime(when, usegmt=True)}, 2) <= 30


def test_past_date_is_no_wait():
    assert retry_delay({"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}, 2) == 0
