"""Tests for the contribution-norms adapter (T3.8) — offline & deterministic; `requests` mocked.

Per the DEVPLAN todo: mock fetch -> profile has template + sign-off flag + title pattern +
N exemplars.
"""

import json

import pytest

from src import pr_profile

pytestmark = pytest.mark.m3


class _FakeResponse:
    def __init__(self, status_code, text="", json_data=None):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data

    def json(self):
        return self._json_data


# --------------------------------------------------------------------- _fetch_file


def test_fetch_file_returns_content_on_200(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pr_profile.requests, "get", lambda *a, **k: _FakeResponse(200, text="# Contributing\n")
    )
    assert pr_profile._fetch_file("o/r", "CONTRIBUTING.md") == "# Contributing\n"


def test_fetch_file_returns_none_on_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr_profile.requests, "get", lambda *a, **k: _FakeResponse(404))
    assert pr_profile._fetch_file("o/r", "CONTRIBUTING.md") is None


def test_fetch_file_returns_none_on_request_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*a, **k):
        raise pr_profile.requests.RequestException("timeout")

    monkeypatch.setattr(pr_profile.requests, "get", _raise)
    assert pr_profile._fetch_file("o/r", "CONTRIBUTING.md") is None


def test_fetch_first_existing_tries_paths_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def _fake_get(url, **k):
        calls.append(url)
        if url.endswith("CONTRIBUTING.md") and "/.github/" not in url:
            return _FakeResponse(404)
        return _FakeResponse(200, text="found")

    monkeypatch.setattr(pr_profile.requests, "get", _fake_get)

    result = pr_profile._fetch_first_existing(
        "o/r", ("CONTRIBUTING.md", ".github/CONTRIBUTING.md", "docs/CONTRIBUTING.md")
    )

    assert result == "found"
    assert len(calls) == 2  # first path missed, second hit -- third never tried


# --------------------------------------------------------------------- _mentions_dco


def test_mentions_dco_detects_signed_off_by() -> None:
    assert pr_profile._mentions_dco("Please include a Signed-off-by line.", None) is True


def test_mentions_dco_detects_dco_keyword() -> None:
    assert pr_profile._mentions_dco(None, "We require DCO compliance.") is True


def test_mentions_dco_false_when_absent() -> None:
    assert pr_profile._mentions_dco("Just write good code.", None) is False


def test_mentions_dco_false_when_both_none() -> None:
    assert pr_profile._mentions_dco(None, None) is False


# --------------------------------------------------------------------- _infer_title_pattern


def test_infer_title_pattern_detects_common_bracket_prefix() -> None:
    titles = ["[Bugfix] fix x", "[Bugfix] fix y", "[Core] refactor z", "[Bugfix] fix w"]
    assert pr_profile._infer_title_pattern(titles) == "[Bugfix] ..."


def test_infer_title_pattern_none_when_no_clear_pattern() -> None:
    titles = ["fix the bug", "add a feature", "[Core] refactor"]
    assert pr_profile._infer_title_pattern(titles) is None


def test_infer_title_pattern_none_when_empty() -> None:
    assert pr_profile._infer_title_pattern([]) is None


# --------------------------------------------------------------------- _fetch_recent_merged_prs


def test_fetch_recent_merged_prs_shapes_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pr_profile.requests,
        "get",
        lambda *a, **k: _FakeResponse(
            200,
            json_data={
                "items": [
                    {
                        "title": "[Bugfix] fix x",
                        "html_url": "https://github.com/o/r/pull/1",
                        "body": "a" * 2000,
                    }
                ]
            },
        ),
    )

    result = pr_profile._fetch_recent_merged_prs("o/r", n=5)

    assert len(result) == 1
    assert result[0]["title"] == "[Bugfix] fix x"
    assert result[0]["url"] == "https://github.com/o/r/pull/1"
    assert len(result[0]["body"]) == pr_profile._BODY_EXCERPT_CHARS


def test_fetch_recent_merged_prs_empty_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr_profile.requests, "get", lambda *a, **k: _FakeResponse(500))
    assert pr_profile._fetch_recent_merged_prs("o/r", n=5) == []


def test_fetch_recent_merged_prs_empty_on_request_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*a, **k):
        raise pr_profile.requests.RequestException("timeout")

    monkeypatch.setattr(pr_profile.requests, "get", _raise)
    assert pr_profile._fetch_recent_merged_prs("o/r", n=5) == []


# --------------------------------------------------------------------- build_profile


def test_build_profile_assembles_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pr_profile,
        "_fetch_first_existing",
        lambda repo, paths: (
            "Please include Signed-off-by."
            if paths[0].startswith("CONTRIBUTING")
            else "## Description\n"
        ),
    )
    monkeypatch.setattr(
        pr_profile,
        "_fetch_recent_merged_prs",
        lambda repo, *, n: [
            {"title": "[Bugfix] fix x", "url": "https://x/1", "body": "b1"},
            {"title": "[Bugfix] fix y", "url": "https://x/2", "body": "b2"},
        ],
    )

    profile = pr_profile.build_profile("o/r", n_exemplars=2)

    assert profile.repo == "o/r"
    assert profile.contributing == "Please include Signed-off-by."
    assert profile.pr_template == "## Description\n"
    assert profile.requires_dco is True
    assert profile.title_pattern == "[Bugfix] ..."
    assert profile.title_examples == ("[Bugfix] fix x", "[Bugfix] fix y")
    assert len(profile.exemplars) == 2


def test_build_profile_requires_dco_false_when_neither_mentions_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pr_profile, "_fetch_first_existing", lambda repo, paths: None)
    monkeypatch.setattr(pr_profile, "_fetch_recent_merged_prs", lambda repo, *, n: [])

    profile = pr_profile.build_profile("o/r")

    assert profile.requires_dco is False
    assert profile.contributing is None
    assert profile.pr_template is None
    assert profile.exemplars == ()


# --------------------------------------------------------------------- RepoProfile round trip


def test_repo_profile_round_trips_through_json() -> None:
    profile = pr_profile.RepoProfile(
        repo="o/r",
        contributing="text",
        pr_template=None,
        requires_dco=True,
        title_pattern="[Bugfix] ...",
        title_examples=("[Bugfix] fix x",),
        exemplars=({"title": "[Bugfix] fix x", "url": "https://x/1", "body": "b"},),
    )

    round_tripped = pr_profile.RepoProfile.from_dict(json.loads(json.dumps(profile.to_dict())))

    assert round_tripped == profile


# --------------------------------------------------------------------- get_profile caching


def test_get_profile_builds_and_caches_on_first_call(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(
        pr_profile,
        "build_profile",
        lambda repo, **k: calls.append(repo)
        or pr_profile.RepoProfile("o/r", None, None, False, None, (), ()),
    )

    profile = pr_profile.get_profile("o/r", profiles_dir=tmp_path)

    assert profile.repo == "o/r"
    assert len(calls) == 1
    assert (tmp_path / "o-r.json").exists()


def test_get_profile_uses_cache_on_second_call(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        pr_profile,
        "build_profile",
        lambda repo, **k: calls.append(repo)
        or pr_profile.RepoProfile("o/r", None, None, False, None, (), ()),
    )

    pr_profile.get_profile("o/r", profiles_dir=tmp_path)
    pr_profile.get_profile("o/r", profiles_dir=tmp_path)

    assert len(calls) == 1  # second call read the cache, didn't rebuild


def test_get_profile_refresh_forces_rebuild(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        pr_profile,
        "build_profile",
        lambda repo, **k: calls.append(repo)
        or pr_profile.RepoProfile("o/r", None, None, False, None, (), ()),
    )

    pr_profile.get_profile("o/r", profiles_dir=tmp_path)
    pr_profile.get_profile("o/r", profiles_dir=tmp_path, refresh=True)

    assert len(calls) == 2


def test_profile_path_defaults_to_module_level_dir_at_call_time(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mirrors gate.py's `_pr_draft_path` pattern: the `None` sentinel resolves against the
    module-level default *at call time*, so tests can monkeypatch it."""
    monkeypatch.setattr(pr_profile, "_PROFILES_DIR", tmp_path)
    assert pr_profile._profile_path("o/r") == tmp_path / "o-r.json"
