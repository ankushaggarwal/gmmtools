import numpy as np
import pandas as pd
import pytest

from gmmtools import * #GMM_Custom

def make_toy_dataframe(n=200, seed=0):
    """
    Create a small artificial dataset with positive values
    suitable for log-transform-based GMMs.
    """
    rng = np.random.default_rng(seed)

    data = {
        "x": rng.normal(120, 10, n).clip(80, None),
        "y": rng.normal(75, 8, n).clip(40, None),
        "a": rng.lognormal(mean=2.0, sigma=0.3, size=n),
        "b": rng.lognormal(mean=-2.0, sigma=0.4, size=n),
    }

    return pd.DataFrame(data)


@pytest.fixture
def fitted_gmm():
    """
    Fit a GMM_Custom on toy data and return it.
    """
    df = make_toy_dataframe()
    gmm = GMM_Custom(df)
    gmm.fit()
    return gmm


def test_fit_and_basic_attributes(fitted_gmm):
    """
    Basic sanity checks after fitting.
    """
    assert fitted_gmm.gmm is not None
    assert fitted_gmm.ndim == fitted_gmm.data.shape[1]
    assert fitted_gmm.gmm.means_.shape[1] == fitted_gmm.ndim


def test_reduce_single_condition_and_marginalise(fitted_gmm):
    """
    Conditioning + marginalisation should produce a reduced GMM
    with the expected number of dimensions.
    """
    fitted_gmm.reduce(condition="x", x_cond=60, marginalise="b")

    assert fitted_gmm.new_gmm is not None
    assert fitted_gmm.new_gmm.means_.shape[1] == 2  # y, a 


def test_reduce_only_marginalise(fitted_gmm):
    """
    Pure marginalisation should work without conditioning.
    """
    fitted_gmm.reduce(marginalise=["a", "x"])

    assert fitted_gmm.new_gmm is not None
    assert fitted_gmm.new_gmm.means_.shape[1] == 2  # y, b 


def test_multiple_reductions_idempotent(fitted_gmm):
    """
    Repeating the same reduction should not crash or change dimensionality.
    """
    fitted_gmm.reduce(marginalise=["a", "x"])
    dim1 = fitted_gmm.new_gmm.means_.shape[1]

    fitted_gmm.reduce(marginalise=["a", "x"])
    dim2 = fitted_gmm.new_gmm.means_.shape[1]

    assert dim1 == dim2 == 2


def test_reduce_multiple_conditions(fitted_gmm):
    """
    Conditioning on multiple variables should reduce dimensionality correctly.
    """
    fitted_gmm.reduce(
        condition=["x", "y"],
        x_cond=[60, 70],
    )

    assert fitted_gmm.new_gmm.means_.shape[1] == 2  # a, b


def test_update_reduced_gmm_changes_means(fitted_gmm):
    """
    Updating conditioning values should change the reduced means.
    """
    fitted_gmm.reduce(
        condition=["x", "y"],
        x_cond=[60, 70],
    )

    means_before = fitted_gmm.new_gmm.means_.copy()

    fitted_gmm.update_reduced_gmm([40, 80])
    means_after = fitted_gmm.new_gmm.means_

    assert not np.allclose(means_before, means_after)


def test_contour_array_creation(fitted_gmm):
    """
    Contour grid creation should only work in 2D reduced space.
    """
    fitted_gmm.reduce(marginalise=["a", "x"])
    fitted_gmm._create_arrays()

    X, Y, Z = fitted_gmm.get_gmm_contour_data()

    assert X.shape == Y.shape == Z.shape
    assert np.all(np.isfinite(Z))

def test_prob_and_prob_reduced_work(fitted_gmm):
    """
    Probability evaluations should return finite values.
    """
    x = fitted_gmm.data.iloc[:5][["x", "y", "a", "b"]].values
    p = fitted_gmm.prob(x)

    assert p.shape == (5,)
    assert np.all(p > 0)

    fitted_gmm.reduce_to_cols(["x", "y"])
    x2 = fitted_gmm.data.iloc[:5][["x", "y"]].values

    p2 = fitted_gmm.prob_reduced(x2)
    assert p2.shape == (5,)
    assert np.all(p2 > 0)


def test_reduce_to_cols_dimension(fitted_gmm):
    """
    reduce_to_cols should keep exactly the requested columns.
    """
    fitted_gmm.reduce_to_cols(["x", "y"])

    assert fitted_gmm.new_gmm.means_.shape[1] == 2


def test_str_representation(fitted_gmm):
    """
    __str__ should return a non-empty, human-readable summary.
    """
    s = str(fitted_gmm)
    assert isinstance(s, str)
    assert "GMM" in s
    assert len(s) > 20

