import pytest

from tensorforge.presets import (
    get_accelerator_preset,
    get_workload_preset,
    list_accelerator_presets,
    list_workload_presets,
)


def test_workload_preset_names_unique():
    names = list_workload_presets()
    assert len(names) == len(set(names))
    assert len(names) > 0


def test_accelerator_preset_names_unique():
    names = list_accelerator_presets()
    assert len(names) == len(set(names))
    assert len(names) > 0


def test_all_workload_presets_instantiate_cleanly():
    for name in list_workload_presets():
        preset = get_workload_preset(name)
        if preset.kind == "gemm":
            gemm = preset.gemm()
            assert gemm.m > 0 and gemm.n > 0 and gemm.k > 0
        elif preset.kind == "transformer":
            spec = preset.spec()
            assert spec.d_model % spec.num_heads == 0
        elif preset.kind == "conv2d":
            spec = preset.spec()
            assert spec.output_height > 0 and spec.output_width > 0
        elif preset.kind == "cnn":
            workload = preset.workload()
            assert len(workload.layers) > 0
        else:
            pytest.fail(f"unexpected preset kind: {preset.kind}")


def test_all_accelerator_presets_instantiate_cleanly():
    for name in list_accelerator_presets():
        preset = get_accelerator_preset(name)
        assert preset.pe_rows > 0
        assert preset.pe_cols > 0
        assert preset.sram_bytes > 0
        assert preset.clock_hz > 0
        assert preset.bandwidth_bytes_per_second > 0
        preset.pe_array()
        preset.memory_hierarchy()
        preset.hardware_config()
        preset.timing_config()


def test_accelerator_peak_compute_derivation():
    preset = get_accelerator_preset("balanced")
    expected = 2 * preset.pe_rows * preset.pe_cols * preset.clock_hz
    assert preset.peak_compute_flops_per_second == expected
    assert preset.hardware_config().peak_compute_flops_per_second == expected


def test_unknown_workload_preset_raises_clear_error():
    with pytest.raises(ValueError, match="unknown workload preset"):
        get_workload_preset("does-not-exist")


def test_unknown_accelerator_preset_raises_clear_error():
    with pytest.raises(ValueError, match="unknown accelerator preset"):
        get_accelerator_preset("does-not-exist")


@pytest.mark.parametrize("field", ["pe_rows", "pe_cols", "sram_bytes", "clock_hz", "bandwidth_bytes_per_second"])
def test_accelerator_preset_rejects_nonpositive_fields(field):
    from tensorforge.presets import AcceleratorPreset

    kwargs = dict(
        name="bad", pe_rows=16, pe_cols=16, sram_bytes=1024,
        clock_hz=1e9, bandwidth_bytes_per_second=1e9, description="",
    )
    kwargs[field] = 0
    with pytest.raises(ValueError):
        AcceleratorPreset(**kwargs)


def test_compute_heavy_has_more_pes_than_balanced_same_bandwidth():
    balanced = get_accelerator_preset("balanced")
    compute_heavy = get_accelerator_preset("compute_heavy")
    assert compute_heavy.pe_rows * compute_heavy.pe_cols > balanced.pe_rows * balanced.pe_cols
    assert compute_heavy.bandwidth_bytes_per_second == balanced.bandwidth_bytes_per_second


def test_bandwidth_heavy_has_more_bandwidth_than_balanced_same_pe():
    balanced = get_accelerator_preset("balanced")
    bandwidth_heavy = get_accelerator_preset("bandwidth_heavy")
    assert bandwidth_heavy.bandwidth_bytes_per_second > balanced.bandwidth_bytes_per_second
    assert (bandwidth_heavy.pe_rows, bandwidth_heavy.pe_cols) == (balanced.pe_rows, balanced.pe_cols)
