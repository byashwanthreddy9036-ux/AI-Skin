from skin_research.cli import build_parser, main


def test_parser_lists_all_commands():
    parser = build_parser()
    sub = next(a for a in parser._actions if a.dest == "command")
    for cmd in ("doctor", "validate-data", "dataset-info", "assess-quality", "generate-variations", "predict",
                "experiment", "evaluate", "report", "build-site", "demo"):
        assert cmd in sub.choices


def test_experiment_name_override_is_applied(monkeypatch):
    seen = {}

    def fake(args, cfg):
        seen["name"] = cfg.experiment.name
        seen["raw"] = cfg.raw["experiment"]["name"]
        return 0

    from skin_research import cli

    monkeypatch.setitem(cli.COMMANDS, "doctor", (fake, "test"))
    assert main(["--experiment-name", "trial_x", "doctor"]) == 0
    assert seen == {"name": "trial_x", "raw": "trial_x"}
