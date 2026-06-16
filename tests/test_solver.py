import pytest
import torch
from unittest.mock import MagicMock, patch

from toolsbench.profiler import NullProfiler
from toolsbench.solver.base import SolverObjective
from toolsbench.solver.pnp import PnPSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_objective(**kwargs):
    defaults = dict(
        measurement=torch.zeros(1, 1, 8, 8),
        physics=MagicMock(),
        ground_truth_shape=torch.Size([1, 1, 8, 8]),
        num_operators=1,
    )
    defaults.update(kwargs)
    return SolverObjective(**defaults)


def _make_solver(**attrs):
    """Return a PnPSolver with attributes set directly, bypassing benchopt init."""
    solver = PnPSolver.__new__(PnPSolver)
    defaults = dict(
        name="test-solver",
        problem=_make_objective(),
        device=torch.device("cpu"),
        profiler=NullProfiler(),
        distributed_mode=False,
        ctx=None,
        norm_strategy="clip",
        denoiser_lambda_relaxation=None,
        denoiser_sigma=0.05,
        step_size=None,
        step_size_scale=0.99,
        denoiser="drunet",
        patch_size=128,
        overlap=32,
        max_batch_size=0,
        distribute_denoiser=False,
        distribute_physics=False,
        init_method="pseudo_inverse",
        reconstruction=torch.zeros(1, 1, 8, 8),
    )
    defaults.update(attrs)
    for k, v in defaults.items():
        setattr(solver, k, v)
    return solver


def _run_one_iter(solver, prior, data_fidelity, step_size=0.1):
    """Run one PnP iteration using a patched distributed_callback_iter."""
    physics = MagicMock()
    measurements = torch.zeros_like(solver.reconstruction)
    with patch(
        "toolsbench.solver.pnp.distributed_callback_iter",
        return_value=iter([None]),
    ):
        solver._run_pnp_iterations(prior, data_fidelity, physics, measurements, step_size, None)


# ---------------------------------------------------------------------------
# SolverObjective
# ---------------------------------------------------------------------------

class TestSolverObjective:

    def test_construction_with_defaults(self):
        obj = _make_objective(num_operators=3)
        assert obj.num_operators == 3
        assert obj.min_pixel == 0.0
        assert obj.max_pixel == 1.0
        assert obj.weights is None


# ---------------------------------------------------------------------------
# BaseInvprobSolver (via PnPSolver)
# ---------------------------------------------------------------------------

class TestBaseInvprobSolver:

    def test_set_objective_stores_problem(self):
        solver = PnPSolver.__new__(PnPSolver)
        solver.name_prefix = "pnp"
        solver.slurm_nodes = 1
        solver.slurm_ntasks_per_node = 1
        solver.torchrun_nproc_per_node = 1
        measurement = torch.zeros(1, 1, 8, 8)
        with patch("toolsbench.solver.base.setup_distributed_env", return_value=1), \
             patch("toolsbench.solver.base.build_solver_name", return_value="test-solver"):
            solver.set_objective(
                measurement=measurement,
                physics=MagicMock(),
                ground_truth_shape=torch.Size([1, 1, 8, 8]),
                num_operators=1,
            )
        assert isinstance(solver.problem, SolverObjective)
        assert solver.problem.measurement is measurement
        assert solver.name == "test-solver"

    def test_create_denoiser_unknown_raises(self):
        solver = _make_solver(denoiser="unknown_model")
        with pytest.raises(ValueError, match="Unknown denoiser"):
            solver._create_denoiser(torch.device("cpu"))


# ---------------------------------------------------------------------------
# PnPSolver._compute_step_size
# ---------------------------------------------------------------------------

class TestPnPSolverComputeStepSize:

    def test_float_step_size_returned_directly(self):
        solver = _make_solver(step_size=0.5, step_size_scale=0.99)
        assert solver._compute_step_size(MagicMock()) == 0.5

    def test_auto_step_size_calls_helper(self):
        solver = _make_solver(step_size=None, step_size_scale=0.5)
        with patch(
            "toolsbench.solver.pnp.compute_step_size_from_operator", return_value=2.0
        ) as mock_fn:
            result = solver._compute_step_size(MagicMock())
        mock_fn.assert_called_once()
        assert result == pytest.approx(1.0)  # 2.0 * 0.5


# ---------------------------------------------------------------------------
# PnPSolver._setup_components
# ---------------------------------------------------------------------------

class TestPnPSolverSetupComponents:

    def test_returns_prior_and_data_fidelity(self):
        from deepinv.optim.prior import PnP
        from deepinv.optim.data_fidelity import L2

        solver = _make_solver()
        with patch.object(solver, "_create_denoiser", return_value=MagicMock()):
            prior, data_fidelity = solver._setup_components(torch.device("cpu"), ctx=None)

        assert isinstance(prior, PnP)
        assert isinstance(data_fidelity, L2)


# ---------------------------------------------------------------------------
# PnPSolver._run_pnp_iterations
# ---------------------------------------------------------------------------

class TestPnPSolverIterations:

    def test_clip_no_relaxation_clamps_output(self):
        solver = _make_solver(
            norm_strategy="clip",
            denoiser_lambda_relaxation=None,
            reconstruction=torch.full((1, 1, 8, 8), 2.0),
            problem=_make_objective(min_pixel=0.0, max_pixel=1.0),
        )
        prior = MagicMock()
        prior.prox.return_value = torch.full((1, 1, 8, 8), 2.0)
        data_fidelity = MagicMock()
        data_fidelity.grad.return_value = torch.zeros(1, 1, 8, 8)

        _run_one_iter(solver, prior, data_fidelity)

        assert solver.reconstruction.max().item() <= 1.0
        assert solver.reconstruction.min().item() >= 0.0

    def test_dynamic_no_relaxation_rescales(self):
        solver = _make_solver(
            norm_strategy="dynamic",
            denoiser_lambda_relaxation=None,
            reconstruction=torch.full((1, 1, 8, 8), 0.5),
            problem=_make_objective(min_pixel=0.0, max_pixel=1.0),
        )
        prior = MagicMock()
        prior.prox.return_value = torch.full((1, 1, 8, 8), 0.5)
        data_fidelity = MagicMock()
        data_fidelity.grad.return_value = torch.zeros(1, 1, 8, 8)

        _run_one_iter(solver, prior, data_fidelity)

        assert solver.reconstruction.shape == torch.Size([1, 1, 8, 8])
        assert solver.reconstruction.mean().item() == pytest.approx(0.5, abs=1e-4)

    def test_dynamic_with_relaxation_alpha_blends(self):
        # reconstruction=0, grad=0, prox returns 1 → alpha-blend toward 1
        solver = _make_solver(
            norm_strategy="dynamic",
            denoiser_lambda_relaxation=1.0,
            denoiser_sigma=0.05,
            reconstruction=torch.zeros(1, 1, 8, 8),
            problem=_make_objective(min_pixel=0.0, max_pixel=1.0),
        )
        prior = MagicMock()
        prior.prox.return_value = torch.ones(1, 1, 8, 8)
        data_fidelity = MagicMock()
        data_fidelity.grad.return_value = torch.zeros(1, 1, 8, 8)

        step_size = 0.1
        _run_one_iter(solver, prior, data_fidelity, step_size=step_size)

        expected_alpha = (step_size * 1.0) / (1 + step_size * 1.0)
        assert solver.reconstruction.mean().item() == pytest.approx(expected_alpha, abs=1e-4)


# ---------------------------------------------------------------------------
# PnPSolver.get_result
# ---------------------------------------------------------------------------

class TestPnPSolverGetResult:

    def test_contains_reconstruction(self):
        solver = _make_solver(reconstruction=torch.ones(1, 1, 4, 4))
        result = solver.get_result()
        assert "reconstruction" in result
        assert torch.equal(result["reconstruction"], torch.ones(1, 1, 4, 4))

    def test_includes_profiler_metrics(self):
        solver = _make_solver(reconstruction=torch.ones(1, 1, 4, 4))
        solver.profiler = MagicMock()
        solver.profiler.get_current_metrics.return_value = {
            "total_time_sec": 0.5,
            "max_gpu_mb": 100.0,
        }
        result = solver.get_result()
        assert result["total_time_sec"] == 0.5
        assert result["max_gpu_mb"] == 100.0
