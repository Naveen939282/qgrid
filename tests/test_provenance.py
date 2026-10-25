"""Tests for experiment provenance collection.

Tests the collect_provenance function from run_noise_experiment.py
without running any experiment.
"""
from __future__ import annotations

import sys
import platform
from unittest.mock import patch, MagicMock

import pytest

# Check if qiskit_aer is importable (needed for version reporting).
try:
    import qiskit_aer  # noqa: F401
    HAS_QISKIT_AER = True
except ImportError:
    HAS_QISKIT_AER = False


@pytest.fixture
def provenance():
    """Collect provenance metadata from run_noise_experiment."""
    from run_noise_experiment import collect_provenance
    return collect_provenance(seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


class TestProvenanceCollection:
    """Tests for collect_provenance function."""

    def test_returns_dict(self, provenance):
        """collect_provenance returns a dict."""
        assert isinstance(provenance, dict)

    def test_git_commit(self, provenance):
        """git_commit is a string (SHA-1 hash or 'unknown')."""
        commit = provenance["git_commit"]
        assert isinstance(commit, str)
        assert commit == "unknown" or len(commit) == 40

    def test_python_version(self, provenance):
        """python_version contains the current Python version string."""
        pv = provenance["python_version"]
        assert isinstance(pv, str)
        assert sys.version in pv

    def test_os_platform(self, provenance):
        """os_platform contains platform information."""
        plat = provenance["os_platform"]
        assert isinstance(plat, str)
        assert len(plat) > 0

    def test_package_versions_keys(self, provenance):
        """package_versions contains all expected packages."""
        pv = provenance["package_versions"]
        assert isinstance(pv, dict)
        expected = [
            "qiskit", "qiskit-aer", "numpy", "scipy",
            "pandas", "matplotlib", "ortools", "pytest",
        ]
        for pkg in expected:
            assert pkg in pv
            assert isinstance(pv[pkg], str)

    def test_package_versions_retrieved(self, provenance):
        """Actual installed versions are captured (not 'unknown')."""
        pv = provenance["package_versions"]
        # qiskit and qiskit-aer should be available in the test env
        assert pv["qiskit"] != "unknown"
        if HAS_QISKIT_AER:
            assert pv["qiskit-aer"] != "unknown"
        else:
            # qiskit-aer compiled extension may fail to load on some Linux
            # distributions (e.g. Ubuntu 26.04); "unknown" is acceptable then
            pytest.skip("qiskit-aer compiled extension unavailable on this platform")

    def test_random_seeds_match(self, provenance):
        """random_seeds contains the seeds passed to the function."""
        rs = provenance["random_seeds"]
        assert isinstance(rs, dict)
        assert rs["seeds"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_random_seeds_description(self, provenance):
        """random_seeds description explains seed ownership."""
        rs = provenance["random_seeds"]
        assert isinstance(rs["description"], str)
        assert len(rs["description"]) > 20
        # Must mention the key components seeds control
        assert "COBYLA" in rs["description"] or "optimization" in rs["description"].lower()
        assert "StatevectorSampler" in rs["description"] or "sampling" in rs["description"].lower()

    def test_aer_config_structure(self, provenance):
        """aer_config contains correct simulator configuration."""
        ac = provenance["aer_config"]
        assert isinstance(ac, dict)
        assert ac["simulator_type"] == "AerSimulator"
        assert isinstance(ac["basis_gates"], list)
        assert "id" in ac["basis_gates"]
        assert "rz" in ac["basis_gates"]
        assert "rzz" in ac["basis_gates"]
        assert "cx" in ac["basis_gates"]
        assert ac["noise_model_type"] == "depolarizing"
        assert isinstance(ac["noise_targets"], dict)
        assert "single_qubit" in ac["noise_targets"]
        assert "two_qubit" in ac["noise_targets"]
        assert ac["optimization_level"] == 0

    def test_aer_seed_description(self, provenance):
        """aer_config documents that seeds are per-trial."""
        ac = provenance["aer_config"]
        assert "seed_transpiler" in ac
        assert "seed_simulator" in ac
        assert "per-trial" in ac["seed_transpiler"]
        assert "per-trial" in ac["seed_simulator"]

    def test_optimization_config(self, provenance):
        """optimization_config contains optimizer and sampler info."""
        oc = provenance["optimization_config"]
        assert isinstance(oc, dict)
        assert oc["optimizer"] == "COBYLA"
        assert "StatevectorSampler" in oc["sampler_for_optimization"]
        assert "final Aer sampling" in oc["noise_applied"]

    def test_does_not_crash_without_git(self):
        """collect_provenance does not crash even if git is unavailable."""
        from run_noise_experiment import collect_provenance
        result = collect_provenance(seeds=[0])
        assert "git_commit" in result
        assert isinstance(result["git_commit"], str)

    def test_accepts_different_seeds(self):
        """collect_provenance works with different seed lists."""
        from run_noise_experiment import collect_provenance
        result = collect_provenance(seeds=[42, 99])
        assert result["random_seeds"]["seeds"] == [42, 99]

    def test_accepts_empty_seeds(self):
        """collect_provenance works with empty seed list."""
        from run_noise_experiment import collect_provenance
        result = collect_provenance(seeds=[])
        assert result["random_seeds"]["seeds"] == []


class TestGitWorkingTreeDirty:
    """Tests for git_working_tree_dirty field using mocked subprocess."""

    def test_key_exists(self, provenance):
        """git_working_tree_dirty field is present in provenance."""
        assert "git_working_tree_dirty" in provenance

    @patch("subprocess.run")
    def test_clean_working_tree(self, mock_run):
        """Empty git status output → dirty=False (clean tree)."""
        rev_parse = MagicMock(returncode=0, stdout="abc123" * 7)
        status = MagicMock(returncode=0, stdout="")
        mock_run.side_effect = [rev_parse, status]

        from run_noise_experiment import collect_provenance
        result = collect_provenance(seeds=[0])

        assert result["git_commit"] == "abc123" * 7
        assert result["git_working_tree_dirty"] is False

    @patch("subprocess.run")
    def test_modified_tracked_files(self, mock_run):
        """Modified tracked files in git status → dirty=True."""
        rev_parse = MagicMock(returncode=0, stdout="abc123" * 7)
        status = MagicMock(returncode=0, stdout=" M run_noise_experiment.py\n")
        mock_run.side_effect = [rev_parse, status]

        from run_noise_experiment import collect_provenance
        result = collect_provenance(seeds=[0])

        assert result["git_working_tree_dirty"] is True

    @patch("subprocess.run")
    def test_untracked_files_detected(self, mock_run):
        """Untracked files appear in git status --porcelain → dirty=True.

        This verifies that using `git status --porcelain` (rather than
        `git diff --quiet`) correctly detects untracked files, which
        `git diff --quiet` would miss.
        """
        rev_parse = MagicMock(returncode=0, stdout="abc123" * 7)
        # Only an untracked file — git diff --quiet would miss this
        status = MagicMock(returncode=0, stdout="?? new_untracked_file.py\n")
        mock_run.side_effect = [rev_parse, status]

        from run_noise_experiment import collect_provenance
        result = collect_provenance(seeds=[0])

        assert result["git_working_tree_dirty"] is True

    @patch("subprocess.run")
    def test_git_unavailable(self, mock_run):
        """When git command raises FileNotFoundError, dirty='unknown'."""
        mock_run.side_effect = FileNotFoundError(
            2, "No such file or directory: 'git'"
        )

        from run_noise_experiment import collect_provenance
        result = collect_provenance(seeds=[0])

        assert result["git_commit"] == "unknown"
        assert result["git_working_tree_dirty"] == "unknown"

    @patch("subprocess.run")
    def test_git_status_fails(self, mock_run):
        """When git status returns non-zero, dirty='unknown' (not False)."""
        rev_parse = MagicMock(returncode=0, stdout="abc123" * 7)
        # git status fails — should not claim tree is clean
        status = MagicMock(returncode=1, stdout="", stderr="fatal error")
        mock_run.side_effect = [rev_parse, status]

        from run_noise_experiment import collect_provenance
        result = collect_provenance(seeds=[0])

        assert result["git_commit"] == "abc123" * 7
        assert result["git_working_tree_dirty"] == "unknown"

    def test_provenance_contains_all_keys(self, provenance):
        """Provenance dict contains all expected top-level keys."""
        expected_keys = {
            "git_commit",
            "git_working_tree_dirty",
            "python_version",
            "os_platform",
            "package_versions",
            "random_seeds",
            "aer_config",
            "optimization_config",
        }
        assert expected_keys.issubset(set(provenance.keys()))

    def test_package_versions_all_string_values(self, provenance):
        """All package_version values are strings (str or 'unknown')."""
        pv = provenance["package_versions"]
        for pkg, version in pv.items():
            assert isinstance(pkg, str)
            assert isinstance(version, str)


class TestVersionHandling:
    """Tests for graceful handling of version metadata retrieval failures."""

    def test_get_version_returns_unknown_on_import_error(self):
        """_get_version returns 'unknown' for unimportable modules."""
        from run_noise_experiment import _get_version
        assert _get_version("nonexistent_module_xyz_12345") == "unknown"

    def test_get_version_returns_unknown_on_non_import_error(self):
        """_get_version returns 'unknown' for modules that raise non-ImportError.

        Some packages may raise ValueError, RuntimeError, etc. during import
        (e.g., due to native library issues). _get_version should handle all.
        """
        from run_noise_experiment import _get_version
        # Mock __import__ to raise ValueError (not ImportError)
        with patch("builtins.__import__", side_effect=ValueError("Simulated error")):
            result = _get_version("qiskit")
            assert result == "unknown"

    def test_get_version_returns_actual_version(self):
        """_get_version returns the actual __version__ for installed packages."""
        from run_noise_experiment import _get_version
        result = _get_version("numpy")
        assert result != "unknown"
        assert isinstance(result, str)

    def test_collect_provenance_handles_version_failure(self):
        """collect_provenance survives when a package raises a non-ImportError.

        qiskit-aer on some platforms may raise ImportError (caught by both
        _safe_version and _get_version). Other packages might raise
        ValueError or RuntimeError. collect_provenance must not crash.
        """
        import builtins
        real_import = builtins.__import__

        def patched_import(name, *args, **kwargs):
            if name == "qiskit_aer":
                raise RuntimeError("Simulated native library failure")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=patched_import):
            from run_noise_experiment import collect_provenance
            result = collect_provenance(seeds=[0])

        assert result["package_versions"]["qiskit-aer"] == "unknown"
