from __future__ import annotations

import hashlib
import json

import pytest


@pytest.fixture
def scoring_inputs(tmp_path, monkeypatch):
    from scripts import score_cmrc2018_official

    def write(name, text):
        path = tmp_path / name
        path.write_bytes(text.encode("utf-8"))
        return path

    dataset = write(
        "dev.json",
        '{"data": [{"paragraphs": [{"qas": [{"id": "q1"}, {"id": "q2"}]}]}]}',
    )
    scorer = write(
        "evaluate.py",
        """\
from types import SimpleNamespace

nltk = SimpleNamespace(__version__="fixture")

def evaluate(ground_truth, predictions):
    n = len(ground_truth["data"][0]["paragraphs"][0]["qas"])
    if predictions["q1"] == "  原始回答\\n":
        return 100 / 3, 25.0, n, n - len(predictions)
    return 100 / 9, 0.0, n, n - len(predictions)
""",
    )
    native = write("native.json", json.dumps({"q1": "  原始回答\n", "q2": ""}))
    scd = write("scd.json", json.dumps({"q1": "另一个回答"}))
    out = tmp_path / "metrics.json"

    monkeypatch.setattr(score_cmrc2018_official, "DEV_N", 2)
    monkeypatch.setattr(score_cmrc2018_official, "DEV_SHA256", hashlib.sha256(dataset.read_bytes()).hexdigest())
    monkeypatch.setattr(score_cmrc2018_official, "SCORER_SHA256", hashlib.sha256(scorer.read_bytes()).hexdigest())
    args = [
        "--dataset", str(dataset),
        "--scorer", str(scorer),
        "--native-predictions", str(native),
        "--scd-predictions", str(scd),
        "--out", str(out),
    ]
    return score_cmrc2018_official, args, dataset, scorer, scd, out


def test_cli_preserves_official_scores_and_counts(scoring_inputs, capsys):
    script, args, dataset, scorer, _, out = scoring_inputs
    script.main(args)

    result = json.loads(out.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == result
    assert result["native"] == {"F1": 100 / 3, "EM": 25.0, "N": 2, "SKIP": 0}
    assert result["scd"] == {"F1": 100 / 9, "EM": 0.0, "N": 2, "SKIP": 1}
    assert result["delta"] == {"F1": 100 / 9 - 100 / 3, "EM": -25.0}
    assert result["dataset"]["sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert result["scorer"]["sha256"] == hashlib.sha256(scorer.read_bytes()).hexdigest()


@pytest.mark.parametrize("field", ["dataset", "scorer"])
def test_cli_rejects_unpinned_inputs(scoring_inputs, capsys, field):
    script, args, dataset, scorer, _, out = scoring_inputs
    path = {"dataset": dataset, "scorer": scorer}[field]
    path.write_bytes(path.read_bytes() + b"\nraise RuntimeError('unverified scorer executed')\n")

    with pytest.raises(SystemExit, match="2"):
        script.main(args)

    error = capsys.readouterr().err
    assert field in error.lower()
    assert "SHA-256" in error
    assert not out.exists()


@pytest.mark.parametrize(
    ("contents", "message"),
    [('{"q3": "foreign"}', "unknown CMRC2018 qid"), ('{"q1": "first", "q1": "second"}', "duplicate JSON key")],
)
def test_cli_rejects_invalid_prediction_membership(scoring_inputs, capsys, contents, message):
    script, args, _, _, scd, out = scoring_inputs
    scd.write_text(contents, encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        script.main(args)

    assert message in capsys.readouterr().err
    assert not out.exists()
