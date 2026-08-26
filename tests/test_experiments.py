import json
import os

import pytest

from tensorforge.experiments import SCHEMA_VERSION, ExperimentSpec, run_experiment


def make_spec(**overrides):
    defaults = dict(
        name="test-experiment",
        workload_preset="gemm_tiny",
        accelerator_preset="balanced",
        tile_m_values=(32, 64),
        tile_n_values=(32, 64),
        tile_k_values=(32, 64),
    )
    defaults.update(overrides)
    return ExperimentSpec(**defaults)


# --- ExperimentSpec validation -------------------------------------------------

def test_rejects_empty_name():
    with pytest.raises(ValueError):
        make_spec(name="")


@pytest.mark.parametrize("field", ["tile_m_values", "tile_n_values", "tile_k_values"])
def test_rejects_empty_tile_values(field):
    with pytest.raises(ValueError):
        make_spec(**{field: ()})


def test_rejects_unknown_schedule_name():
    with pytest.raises(ValueError):
        make_spec(schedule_names=("not-a-schedule",))


def test_rejects_empty_schedule_names():
    with pytest.raises(ValueError):
        make_spec(schedule_names=())


# --- run_experiment across workload kinds --------------------------------------

@pytest.mark.parametrize(
    "workload_preset,expected_kind",
    [
        ("gemm_tiny", "gemm"),
        ("transformer_small", "transformer"),
        ("conv_pointwise", "conv2d"),
        ("cnn_like_small", "cnn"),
    ],
)
def test_run_experiment_all_workload_kinds(workload_preset, expected_kind):
    spec = make_spec(workload_preset=workload_preset)
    result = run_experiment(spec)

    assert result.workload_kind == expected_kind
    assert result.primary_metrics["total_flops"] > 0
    assert result.primary_metrics["total_dram_bytes"] > 0
    assert result.primary_metrics["perfect_overlap_time_seconds"] > 0
    assert len(result.selected_mappings) > 0


def test_unknown_workload_preset_raises():
    spec = make_spec(workload_preset="does-not-exist")
    with pytest.raises(ValueError):
        run_experiment(spec)


def test_unknown_accelerator_preset_raises():
    spec = make_spec(accelerator_preset="does-not-exist")
    with pytest.raises(ValueError):
        run_experiment(spec)


def test_infeasible_search_raises_clean_error():
    # gemm_large_square (1024^3) with tiny tile candidates and tiny SRAM (small preset)
    # combined with tiles too large to fit -- force infeasibility with a huge tile.
    spec = make_spec(
        workload_preset="gemm_large_square", accelerator_preset="small",
        tile_m_values=(1024,), tile_n_values=(1024,), tile_k_values=(1024,),
    )
    with pytest.raises(ValueError):
        run_experiment(spec)


# --- JSON result format ---------------------------------------------------------

def test_json_schema_version():
    result = run_experiment(make_spec())
    data = result.to_dict()
    assert data["schema_version"] == SCHEMA_VERSION == 1


def test_json_top_level_keys():
    result = run_experiment(make_spec())
    data = result.to_dict()
    for key in ("schema_version", "name", "workload", "accelerator", "search",
                "primary_metrics", "selected_mappings", "limitations"):
        assert key in data


def test_json_no_timestamp_fields():
    result = run_experiment(make_spec())
    data = result.to_dict()
    serialized = json.dumps(data)
    for forbidden in ("timestamp", "wall_clock", "datetime", "created_at"):
        assert forbidden not in serialized.lower()


def test_json_is_valid_and_round_trips():
    result = run_experiment(make_spec())
    text = result.to_json()
    parsed = json.loads(text)
    assert parsed == result.to_dict()


def test_json_preserves_numeric_precision():
    result = run_experiment(make_spec())
    data = result.to_dict()
    # perfect_overlap_time_seconds should remain a float, not a rounded string.
    value = data["primary_metrics"]["perfect_overlap_time_seconds"]
    assert isinstance(value, float)


# --- determinism -----------------------------------------------------------------

def test_repeated_run_is_byte_identical():
    spec = make_spec()
    result1 = run_experiment(spec)
    result2 = run_experiment(spec)
    assert result1.to_json() == result2.to_json()


@pytest.mark.parametrize("workload_preset", ["transformer_small", "conv_pointwise", "cnn_like_small"])
def test_repeated_run_deterministic_all_kinds(workload_preset):
    spec = make_spec(workload_preset=workload_preset)
    result1 = run_experiment(spec)
    result2 = run_experiment(spec)
    assert result1.to_json() == result2.to_json()


# --- save() overwrite protection --------------------------------------------------

def test_save_creates_file(tmp_path):
    result = run_experiment(make_spec())
    path = str(tmp_path / "result.json")
    result.save(path)
    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert data["schema_version"] == 1


def test_save_refuses_overwrite_without_force(tmp_path):
    result = run_experiment(make_spec())
    path = str(tmp_path / "result.json")
    result.save(path)
    with pytest.raises(FileExistsError):
        result.save(path)


def test_save_overwrites_with_force(tmp_path):
    result = run_experiment(make_spec())
    path = str(tmp_path / "result.json")
    result.save(path)
    result.save(path, force=True)  # must not raise
    assert os.path.exists(path)


# --- schedule restriction -----------------------------------------------------

def test_restricting_schedules_only_uses_requested_schedule():
    spec = make_spec(schedule_names=("a-resident",))
    result = run_experiment(spec)
    assert result.selected_mappings[0]["schedule"] == "a-resident"
