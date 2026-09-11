"""Filter names used by the command-line runners."""

from filters.EKF import EKF
from filters.UKF import UKF
from filters.PF import PF
from filters.ESKF import ESKF
from filters.InEKF import InEKF

FILTER_REGISTRY = {
    "ekf": EKF,
    "quat_ekf": EKF,
    "ukf": UKF,
    "quat_ukf": UKF,
    "pf": PF,
    "quat_pf": PF,
    "eskf": ESKF,
    "inekf": InEKF,
}

FILTER_CONFIG_KEY = {
    "ekf": "quaternion_ekf_15d",
    "quat_ekf": "quaternion_ekf_15d",
    "ukf": "quaternion_manifold_ukf_15d",
    "quat_ukf": "quaternion_manifold_ukf_15d",
    "pf": "quaternion_manifold_pf_15d",
    "quat_pf": "quaternion_manifold_pf_15d",
    "eskf": "rotation_eskf_15d",
    "inekf": "Hoon_invariant_kalman_analytic_15d",
}

CANONICAL_NAME = {
    "ekf": "ekf",
    "quat_ekf": "ekf",
    "ukf": "ukf",
    "quat_ukf": "ukf",
    "pf": "pf",
    "quat_pf": "pf",
    "eskf": "eskf",
    "inekf": "inekf",
}


def get_filter_class(name: str):
    key = name.lower()
    if key not in FILTER_REGISTRY:
        valid = ", ".join(sorted(FILTER_REGISTRY))
        raise KeyError(f"Unknown filter '{name}'. Valid filters: {valid}")
    return FILTER_REGISTRY[key]


def get_filter_config_key(name: str) -> str:
    return FILTER_CONFIG_KEY[name.lower()]


def get_canonical_name(name: str) -> str:
    return CANONICAL_NAME[name.lower()]
