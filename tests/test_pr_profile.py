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
        self.encoding = None
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


def test_infer_title_pattern_detects_bracket_categories() -> None:
    """Regression: an earlier version looked for one single repeated literal prefix and
    returned None for a repo (vllm-project/vllm) that uses several different category tags --
    the real, common convention. This checks the fixed behavior: report that bracket tags are
    used at all, plus which categories were actually observed."""
    titles = ["[Bugfix] fix x", "[Bugfix] fix y", "[Core] refactor z", "[Bugfix] fix w"]
    result = pr_profile._infer_title_pattern(titles)
    assert result is not None
    assert result.startswith("[Category] ...")
    assert "[Bugfix]" in result
    assert "[Core]" in result


def test_infer_title_pattern_none_when_no_clear_pattern() -> None:
    titles = ["fix the bug", "add a feature", "[Core] refactor"]
    assert pr_profile._infer_title_pattern(titles) is None


def test_infer_title_pattern_none_when_empty() -> None:
    assert pr_profile._infer_title_pattern([]) is None


def test_infer_title_pattern_none_below_minimum_sample_size() -> None:
    """Regression: a 'pattern' inferred from a single title isn't a pattern."""
    assert pr_profile._infer_title_pattern(["[Bugfix] fix x"]) is None
    assert pr_profile._infer_title_pattern(["[Bugfix] fix x", "[Bugfix] fix y"]) is None


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
            {"title": "[Core] fix z", "url": "https://x/3", "body": "b3"},
        ],
    )

    profile = pr_profile.build_profile("o/r", n_exemplars=3)

    assert profile.repo == "o/r"
    assert profile.contributing == "Please include Signed-off-by."
    assert profile.pr_template == "## Description\n"
    assert profile.requires_dco is True
    assert profile.title_pattern is not None
    assert "[Bugfix]" in profile.title_pattern
    assert profile.title_examples == ("[Bugfix] fix x", "[Bugfix] fix y", "[Core] fix z")
    assert len(profile.exemplars) == 3


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
        exemplars=({"title": "[Bugfix] fix x", "url": "https://x/1", "body": "b"},),
    )

    round_tripped = pr_profile.RepoProfile.from_dict(json.loads(json.dumps(profile.to_dict())))

    assert round_tripped == profile


def test_repo_profile_derived_properties_stay_consistent_with_stored_fields() -> None:
    """Regression: requires_dco/title_pattern/title_examples used to be independently stored
    fields that could drift out of sync with contributing/pr_template/exemplars (e.g. a
    hand-edited cache file). They're properties now, computed fresh from the stored fields --
    this pins that they can never disagree."""
    profile = pr_profile.RepoProfile(
        repo="o/r",
        contributing="Please include a Signed-off-by line.",
        pr_template=None,
        exemplars=(
            {"title": "[Bugfix] fix x", "url": "https://x/1", "body": "b"},
            {"title": "[Bugfix] fix y", "url": "https://x/2", "body": "b"},
            {"title": "[Core] fix z", "url": "https://x/3", "body": "b"},
        ),
    )

    assert profile.requires_dco is True
    assert profile.title_examples == ("[Bugfix] fix x", "[Bugfix] fix y", "[Core] fix z")
    assert profile.title_pattern is not None


# --------------------------------------------------------------------- get_profile caching


def test_get_profile_builds_and_caches_on_first_call(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(
        pr_profile,
        "build_profile",
        lambda repo, **k: calls.append(repo) or pr_profile.RepoProfile("o/r", None, None, ()),
    )

    profile = pr_profile.get_profile("o/r", profiles_dir=tmp_path, n_exemplars=0)

    assert profile.repo == "o/r"
    assert len(calls) == 1
    assert (tmp_path / "o-r.json").exists()


def test_get_profile_uses_cache_on_second_call(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        pr_profile,
        "build_profile",
        lambda repo, **k: calls.append(repo) or pr_profile.RepoProfile("o/r", None, None, ()),
    )

    pr_profile.get_profile("o/r", profiles_dir=tmp_path, n_exemplars=0)
    pr_profile.get_profile("o/r", profiles_dir=tmp_path, n_exemplars=0)

    assert len(calls) == 1  # second call read the cache, didn't rebuild


def test_get_profile_rebuilds_when_cache_has_fewer_exemplars_than_requested(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache built with a smaller n_exemplars must not silently short-change a later caller
    that asks for more."""
    calls = []
    monkeypatch.setattr(
        pr_profile,
        "build_profile",
        lambda repo, *, n_exemplars: calls.append(n_exemplars)
        or pr_profile.RepoProfile(
            "o/r",
            None,
            None,
            tuple({"title": "t", "url": "u", "body": "b"} for _ in range(n_exemplars)),
        ),
    )

    pr_profile.get_profile("o/r", profiles_dir=tmp_path, n_exemplars=2)
    profile = pr_profile.get_profile("o/r", profiles_dir=tmp_path, n_exemplars=5)

    assert calls == [2, 5]  # second call rebuilt, didn't settle for the smaller cache
    assert len(profile.exemplars) == 5


def test_get_profile_refresh_forces_rebuild(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        pr_profile,
        "build_profile",
        lambda repo, **k: calls.append(repo) or pr_profile.RepoProfile("o/r", None, None, ()),
    )

    pr_profile.get_profile("o/r", profiles_dir=tmp_path, n_exemplars=0)
    pr_profile.get_profile("o/r", profiles_dir=tmp_path, n_exemplars=0, refresh=True)

    assert len(calls) == 2


def test_profile_path_defaults_to_module_level_dir_at_call_time(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mirrors gate.py's `_pr_draft_path` pattern: the `None` sentinel resolves against the
    module-level default *at call time*, so tests can monkeypatch it."""
    monkeypatch.setattr(pr_profile, "_PROFILES_DIR", tmp_path)
    assert pr_profile._profile_path("o/r") == tmp_path / "o-r.json"


# --------------------------------------------------------------------- cache robustness


def test_get_profile_rebuilds_on_corrupted_cache(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: a truncated/corrupted cache file (an interrupted write, or an incompatible
    schema from an older version of this module) must degrade to a rebuild, not crash whatever
    called get_profile."""
    path = tmp_path / "o-r.json"
    path.write_text("{not valid json")
    calls = []
    monkeypatch.setattr(
        pr_profile,
        "build_profile",
        lambda repo, **k: calls.append(repo) or pr_profile.RepoProfile("o/r", None, None, ()),
    )

    profile = pr_profile.get_profile("o/r", profiles_dir=tmp_path, n_exemplars=0)

    assert profile.repo == "o/r"
    assert len(calls) == 1


def test_get_profile_rebuilds_on_incompatible_schema(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache file from an older schema version (missing a now-required key) must also
    degrade to a rebuild rather than raising KeyError."""
    path = tmp_path / "o-r.json"
    path.write_text(json.dumps({"repo": "o/r"}))  # missing contributing/pr_template/exemplars
    calls = []
    monkeypatch.setattr(
        pr_profile,
        "build_profile",
        lambda repo, **k: calls.append(repo) or pr_profile.RepoProfile("o/r", None, None, ()),
    )

    profile = pr_profile.get_profile("o/r", profiles_dir=tmp_path, n_exemplars=0)

    assert profile.repo == "o/r"
    assert len(calls) == 1


def test_write_cached_profile_is_atomic(tmp_path) -> None:
    """The cache write goes through a .tmp sibling + rename, not a direct write -- no .tmp file
    should be left behind after a successful write."""
    path = tmp_path / "o-r.json"
    profile = pr_profile.RepoProfile("o/r", None, None, ())

    pr_profile._write_cached_profile(profile, path)

    assert path.exists()
    assert not path.with_suffix(".json.tmp").exists()
    assert pr_profile.RepoProfile.from_dict(json.loads(path.read_text())) == profile


# --------------------------------------------------------------------- charset handling


def test_fetch_file_forces_utf8_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: GitHub doesn't always send a charset in Content-Type for raw content;
    `requests` would otherwise guess (historically ISO-8859-1 for text/*), mis-decoding
    non-ASCII text. `_fetch_file` must force UTF-8 explicitly."""
    response = _FakeResponse(200, text="café")
    response.encoding = "ISO-8859-1"  # what requests might have guessed
    monkeypatch.setattr(pr_profile.requests, "get", lambda *a, **k: response)

    pr_profile._fetch_file("o/r", "CONTRIBUTING.md")

    assert response.encoding == "utf-8"


# --------------------------------------------------------------------- merged-PR ordering


def test_fetch_recent_merged_prs_sorts_by_closed_at_not_updated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: GitHub's search API only supports sort=updated, not sort=merged -- a PR
    merged long ago but recently touched (a bot comment, a label) would otherwise outrank a
    genuinely recent merge. Re-sort client-side by closed_at (the available proxy for merge
    time) before truncating to n."""
    monkeypatch.setattr(
        pr_profile.requests,
        "get",
        lambda *a, **k: _FakeResponse(
            200,
            json_data={
                "items": [
                    {
                        "title": "old PR, recently touched",
                        "html_url": "https://x/1",
                        "body": "",
                        "closed_at": "2020-01-01T00:00:00Z",
                    },
                    {
                        "title": "genuinely recent merge",
                        "html_url": "https://x/2",
                        "body": "",
                        "closed_at": "2026-01-01T00:00:00Z",
                    },
                ]
            },
        ),
    )

    result = pr_profile._fetch_recent_merged_prs("o/r", n=1)

    assert len(result) == 1
    assert result[0]["title"] == "genuinely recent merge"


def test_fetch_recent_merged_prs_overfetches_for_client_side_sort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_params = {}

    def _fake_get(url, *, params, **k):
        captured_params.update(params)
        return _FakeResponse(200, json_data={"items": []})

    monkeypatch.setattr(pr_profile.requests, "get", _fake_get)

    pr_profile._fetch_recent_merged_prs("o/r", n=5)

    assert int(captured_params["per_page"]) == 5 * pr_profile._OVERFETCH_FACTOR
