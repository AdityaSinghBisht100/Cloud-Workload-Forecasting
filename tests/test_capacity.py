import copy

import numpy as np
import pandas as pd
import pytest

from cwfc.capacity import capacity_decisions, load_policy, summarize_policy, validate_evidence
from cwfc.uncertainty import conformal_quantile, cpu_bounds


def inputs():
    histories = pd.DataFrame({"timestamp": np.arange(12)*300, "fleet": np.arange(12, dtype=float)+10})
    rows = []
    for origin in [3, 5, 7]:
        for step in [1, 2, 3, 4]:
            index = origin+step-1
            rows.append({"fold": origin, "origin": origin, "step": step, "model": "fixed",
                         "target": "fleet", "timestamp": index*300,
                         "y": histories.fleet.iloc[index], "yhat": histories.fleet.iloc[origin-1]})
    settings = {"segments": {"fleet": {"under_cost": 3., "over_cost": 1., "margin": .1}},
                "cost_curve_levels": [.5, .75, .9], "fixed_headroom_pp": [0., .25],
                "peak_trigger": {"slope_pp_per_step": .5, "extra_headroom_pp": .25}}
    return histories, pd.DataFrame(rows), settings


def test_finite_sample_rank_and_physical_bounds():
    assert conformal_quantile([1, 2, 3], .75) == 3
    assert np.isinf(conformal_quantile(np.arange(7), .9))
    assert conformal_quantile(np.arange(9), .9) == 8
    empty = cpu_bounds(10, [], .9)
    assert empty["upper_cpu"] == empty["interval_upper_cpu"] == 100
    assert empty["lower_cpu"] == 0 and not empty["upper_supported"]
    bounds = cpu_bounds(10, [-3, -2, -1], .5)
    assert bounds["upper_cpu"] == 8  # signed correction, not absolute-error headroom
    assert bounds["lower_cpu"] == 8 and bounds["interval_upper_cpu"] == 12
    assert cpu_bounds(99, [10]*9, .9)["upper_cpu"] == 100


@pytest.mark.parametrize("scores,level", [([np.nan], .9), ([np.inf], .9), ([[1]], .9),
                                          ([], 0), ([], 1), ([], np.nan)])
def test_reject_invalid_calibration(scores, level):
    with pytest.raises(ValueError):
        conformal_quantile(scores, level)


def test_only_matured_same_step_residuals_and_future_invariance():
    histories, evidence, settings = inputs()
    decisions = capacity_decisions(evidence, evidence, histories, settings)
    # At origin 5, previous origin's step 1/2 labels arrived, but step 3/4 did not.
    current = decisions.loc[decisions.origin.eq(5) & decisions.policy.eq("quantile_0.5")].set_index("step")
    assert current.calibration_n.to_list() == [1, 1, 0, 0]
    assert current.loc[2, "latest_calibration_timestamp"] == 1200
    assert decisions.latest_calibration_timestamp.dropna().le(
        decisions.loc[decisions.latest_calibration_timestamp.notna(), "cutoff_timestamp"]).all()
    changed_history, changed = histories.copy(), evidence.copy()
    changed_history.loc[5:, "fleet"] += 30
    changed.loc[changed.timestamp.ge(1500), "y"] += 30
    changed.loc[changed.origin.ge(5), "yhat"] += 20
    # Score only origin 5 using unchanged current point forecasts; future evidence
    # and labels cannot change its bounds, counts, slope or capacity.
    cases = evidence.loc[evidence.origin.eq(5)].drop(columns="y")
    a = capacity_decisions(cases, evidence, histories, settings)
    b = capacity_decisions(cases, changed, changed_history, settings)
    pd.testing.assert_frame_equal(a, b)


def test_margin_cost_ratio_and_slope_trigger():
    histories, evidence, settings = inputs()
    decisions = capacity_decisions(evidence, evidence, histories, settings)
    default = decisions.loc[decisions.policy.eq("cost_ratio_default")]
    assert default.level.eq(.75).all()
    assert np.allclose(default.capacity_cpu, np.minimum(default.upper_cpu*1.1, 100))
    peak = decisions.loc[decisions.policy.eq("cost_ratio_peak_trigger")]
    assert np.allclose(peak.capacity_cpu, np.minimum(default.capacity_cpu.to_numpy()+.25*peak.peak_triggered, 100))
    assert not default.loc[default.origin.eq(3), "peak_triggered"].any()
    assert default.loc[default.origin.ge(5), "peak_triggered"].all()
    fixed = decisions.loc[decisions.policy.eq("fixed_0.25pp")]
    assert np.allclose(fixed.capacity_cpu, fixed.yhat+.25)


def test_cost_coverage_and_pinball_by_hand():
    histories, evidence, settings = inputs()
    data = capacity_decisions(evidence, evidence, histories, settings).iloc[:2].copy()
    data["policy"] = "test"
    data["y"] = [12., 8.]
    data["capacity_cpu"] = [10., 10.]
    data["upper_cpu"] = [10., 10.]
    data["lower_cpu"] = [9., 9.]
    data["interval_upper_cpu"] = [11., 11.]
    data["level"] = .75
    score = summarize_policy(data).iloc[0]
    assert score.under_provision_rate == .5
    assert score.mean_shortfall_pp == score.mean_over_provision_pp == 1
    assert score.mean_cost_proxy == 4  # (3*2+1*2)/2
    assert score.mean_pinball_pp == 1
    assert score.upper_coverage == .5 and score.interval_coverage == 0


@pytest.mark.parametrize("failure", ["duplicates", "label", "timestamp", "nonfinite", "grid", "step"])
def test_evidence_validation(failure):
    histories, evidence, _ = inputs()
    if failure == "duplicates":
        evidence = pd.concat([evidence, evidence.iloc[:1]])
    elif failure == "label":
        evidence.loc[0, "y"] += 1
    elif failure == "timestamp":
        evidence.loc[0, "timestamp"] += 300
    elif failure == "nonfinite":
        evidence.loc[0, "yhat"] = np.nan
    elif failure == "grid":
        histories.loc[0, "timestamp"] += 1
    else:
        evidence["step"] = evidence.step.astype(float)
        evidence.loc[0, "step"] = 1.5
    with pytest.raises(ValueError):
        validate_evidence(evidence, histories)


def test_model_and_target_residuals_are_isolated():
    histories, evidence, settings = inputs()
    extra = evidence.assign(model="other", yhat=0)
    cases = evidence.drop(columns="y")
    a = capacity_decisions(cases, evidence, histories, settings)
    b = capacity_decisions(cases, pd.concat([evidence, extra]), histories, settings)
    pd.testing.assert_frame_equal(a, b)


