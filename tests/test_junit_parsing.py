"""JUnit XML from pytest runs is parsed with defusedxml and fails loudly."""

import textwrap

import pytest

import mcp_server.runs as runs
from mcp_server.runs import JUnitReportError, PytestRunner, parse_junit

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


VALID = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="4">
  <testcase classname="t" name="ok"/>
  <testcase classname="t" name="bad"><failure message="boom">trace</failure></testcase>
  <testcase classname="t" name="err"><error message="oops">tb</error></testcase>
  <testcase classname="t" name="skip"><skipped message="later"/></testcase>
</testsuite></testsuites>
"""

BILLION_LAUGHS = """<?xml version="1.0"?>
<!DOCTYPE lolz [
  <!ENTITY lol "lol">
  <!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
  <!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">
  <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
]>
<testsuites><testsuite><testcase classname="&lol3;" name="x"/></testsuite></testsuites>
"""

EXTERNAL_ENTITY = """<?xml version="1.0"?>
<!DOCTYPE r [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<testsuites><testsuite><testcase classname="&xxe;" name="x"/></testsuite></testsuites>
"""

EXTERNAL_DTD = """<?xml version="1.0"?>
<!DOCTYPE r SYSTEM "http://attacker.invalid/evil.dtd">
<testsuites><testsuite><testcase classname="c" name="x"/></testsuite></testsuites>
"""


def _write(tmp_path, text):
    path = tmp_path / "junit.xml"
    path.write_text(text)
    return path


def test_valid_report_is_counted(tmp_path):
    summary, failures = parse_junit(_write(tmp_path, VALID))
    assert summary == {"total": 4, "passed": 1, "failed": 1, "errors": 1, "skipped": 1}
    assert [(f["test"], f["kind"], f["message"]) for f in failures] == [
        ("t::bad", "failure", "boom"),
        ("t::err", "error", "oops"),
    ]


def test_single_testsuite_root_is_accepted(tmp_path):
    xml = '<testsuite><testcase classname="c" name="n"/></testsuite>'
    summary, _ = parse_junit(_write(tmp_path, xml))
    assert summary["total"] == 1


def test_missing_report_returns_empty(tmp_path):
    assert parse_junit(tmp_path / "absent.xml") == ({}, [])


def test_malformed_report_raises_a_clear_error(tmp_path):
    with pytest.raises(JUnitReportError, match="not well-formed"):
        parse_junit(_write(tmp_path, "<testsuites><testsuite>"))


def test_unexpected_root_element_is_rejected(tmp_path):
    with pytest.raises(JUnitReportError, match="unexpected root element <html>"):
        parse_junit(_write(tmp_path, "<html><body/></html>"))


@pytest.mark.parametrize(
    "xml,exc_name",
    [
        (BILLION_LAUGHS, "EntitiesForbidden"),
        (EXTERNAL_ENTITY, "EntitiesForbidden"),
    ],
)
def test_hostile_entities_are_refused_before_expansion(tmp_path, xml, exc_name):
    with pytest.raises(JUnitReportError, match="unsafe XML") as err:
        parse_junit(_write(tmp_path, xml))
    assert exc_name in str(err.value)


def test_external_dtd_is_not_fetched(tmp_path):
    # The DTD URL is unreachable by design: if it were fetched, the parse would
    # hang or fail with a network error instead of returning normally.
    summary, _ = parse_junit(_write(tmp_path, EXTERNAL_DTD))
    assert summary["total"] == 1


def test_oversized_report_is_not_parsed(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "MAX_JUNIT_BYTES", 100)
    with pytest.raises(JUnitReportError, match="over the 100 byte limit"):
        parse_junit(_write(tmp_path, VALID))


# -- through a real pytest run -------------------------------------------------


def _project(tmp_path, conftest=""):
    project = tmp_path / "proj"
    (project / "tests").mkdir(parents=True)
    (project / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    if conftest:
        (project / "conftest.py").write_text(textwrap.dedent(conftest))
    return project


async def test_corrupt_report_keeps_exit_status_and_explains(tmp_path):
    # pytest writes junit.xml in pytest_sessionfinish; this hook runs later and
    # overwrites it with broken XML.
    project = _project(
        tmp_path,
        """
        def pytest_unconfigure(config):
            path = config.option.xmlpath
            if path:
                with open(path, "w") as fh:
                    fh.write("<testsuites><testsuite")
        """,
    )
    record = await PytestRunner(project).run(target="tests")

    assert record.status == "passed"
    assert record.exit_code == 0
    assert record.summary == {}
    assert "not well-formed" in record.report_error
    assert record.to_dict()["report_error"] == record.report_error


async def test_healthy_run_has_no_report_error(tmp_path):
    record = await PytestRunner(_project(tmp_path)).run(target="tests")
    assert record.status == "passed"
    assert record.summary["passed"] == 1
    assert record.report_error is None
