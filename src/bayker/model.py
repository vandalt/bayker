from pathlib import Path

import numpy as np
import yaml
from simpple.distributions import Distribution
from simpple.model import ForwardModel
from simpple.load import parse_parameters, unparse_parameters
from simpple.utils import find_args
from xara.kpo import KPO

from bayker.utils import seppa2radec

MAS_2_RAD = 1 / 1000.0 / 3600.0 * np.pi / 180.0


def forward_single(p: dict, kpo: KPO, *args) -> np.ndarray:
    """Forward model for single object (zeros)

    Dummy forward model passed to simpple when no companions are fitted.
    Has no free parameters. Arguments are for compatibility with the actual KPI model signature.

    :param p: Parameter dictionary, no elements are used
    :param kpo: Kernel phase object
    :return: Array of zeros with length of ``n_kp``
    """
    return np.zeros(kpo.kpi.KPM.shape[0])


def forward_binary(p: dict, kpo: KPO, pos_param: str) -> np.ndarray:
    """Forward model for a binary

    :param p: Parameter dictionary. Should always contain ``cr`` for contrast ratio, as well as the two position parameters (``dra`` and ``ddec``, or ``sep`` and ``pa``). Distances should be in milliarcseconds, PA in degrees.
    :param kpo: Kernel phase object from xara.
    :param pos_param: String giving the position parametrization. Should be ``"radec"`` or ``"seppa"``.
    :return: Array of kernel phase model values
    """
    uu, vv = kpo.kpi.UVC.T / kpo.CWAVEL
    if pos_param == "seppa":
        dra, ddec = seppa2radec(p["sep"], p["pa"])
    else:
        dra = p["dra"]
        ddec = p["ddec"]
    cr = p["cr"]
    a1 = 1.0
    a2 = cr
    dra_rad = dra * MAS_2_RAD
    ddec_rad = ddec * MAS_2_RAD

    # Handle both scalar and meshgrid inputs efficiently
    if np.ndim(dra_rad) == 0:  # Scalar case
        companion_cvis = a2 * np.exp(
            -1.0j * 2.0 * np.pi * (uu * dra_rad + vv * ddec_rad)
        )
        cvis_binary = (a1 + companion_cvis) / (a1 + a2)
    else:  # meshgrid case
        uu = uu[:, np.newaxis, np.newaxis]
        vv = vv[:, np.newaxis, np.newaxis]
        companion_cvis = a2 * np.exp(
            -1.0j * 2.0 * np.pi * (uu * dra_rad + vv * ddec_rad)
        )
        cvis_binary = (a1 + companion_cvis) / (a1 + a2)
        cvis_binary = cvis_binary.swapaxes(0, 1)
    return kpo.kpi.KPM @ np.angle(cvis_binary)


class KernelModel(ForwardModel):
    """Kernel Phase Model
    `simpple.model.ForwardModel <https://simpple.readthedocs.io/en/stable/api/model.html#simpple.model.ForwardModel>` subclass for RV models.

    :param parameters: Model parameters specified as a dictionary of `simpple.distribution.Distribution <https://simpple.readthedocs.io/en/stable/api/distributions.html>` objects.
    :param kpo: xara.kpo.KPO object storing the kernel phase data and pupil model information
    :param model_type: Whether the forward model should be for a binary or a single source
    :param pos_param: Position parametrization for the binary model ("seppa" or "radec"). Defaults to "seppa".
    """

    kpfits = None
    yaml_file = None

    def __init__(
        self,
        parameters: dict[str, Distribution],
        kpo: KPO,
        model_type: str = "binary",
        pos_param: str = "seppa",
    ):
        super().__init__(parameters)

        # HACK: Add arrays to kpo data
        if not hasattr(kpo, "kp"):
            kpo.kp = np.array(kpo.KPDT).squeeze()
        if not hasattr(kpo, "ekp"):
            kpo.ekp = np.array(kpo.KPSIG).squeeze()
        if not hasattr(kpo, "x"):
            kpo.x = np.arange(kpo.kp.shape[-1])
        self.kpo = kpo
        self.share_sigma = True
        self.model_type = model_type
        self.pos_param = pos_param

        if "sigma" in parameters:
            self.share_sigma = True
        elif "sigma0" in parameters:
            self.share_sigma = False

        expected_params = []
        if self.model_type == "binary":
            expected_params.append("cr")
            if self.pos_param == "seppa":
                expected_params += ["sep", "pa"]
            elif self.pos_param == "radec":
                expected_params += ["dra", "ddec"]
            else:
                raise ValueError(
                    f"Unexpected position parameterization {self.pos_param}. Use 'seppa' or 'radec'."
                )
            self._forward = lambda p: forward_binary(p, self.kpo, self.pos_param)
        elif self.model_type == "single":
            self._forward = lambda p: forward_single(p, self.kpo, self.pos_param)
        else:
            raise ValueError(
                f"Unexpected model type {self.model_type}. Use 'single' or 'binary'."
            )
        if self.kpo.kp.ndim > 1 and self.kpo.kp.shape[0] > 1 and not self.share_sigma:
            for i in range(kpo.kp.shape[0]):
                expected_params.append(f"sigma{i}")
            self._log_likelihood = self._log_likelihood_split
        else:
            expected_params.append("sigma")
            self._log_likelihood = self._log_likelihood_shared

    @classmethod
    def from_kpfits(
        cls: "KernelModel",
        parameters: dict[str, Distribution],
        kpfits: Path | str | list[Path | str],
        model_type: str = "binary",
        pos_param: str = "seppa",
    ) -> "KernelModel":
        """Build KernelModel from one or more KPFITS.

        :param parameters: Dictionary of parameters as ``simpple`` distributions.
        :param kpfits: Path to one or more KPFITS file. A xara KPO will be build from the file(s) directly.
        :param model_type: Model type ("binary" or "single")
        :param pos_param: Position parametrization ("seppa" or "radec")
        """
        if not isinstance(kpfits, (Path, str)) and len(kpfits) > 1:
            kpfits = [str(f) for f in kpfits]
            kpo = KPO.from_kpfits_list(kpfits)
        else:
            if not isinstance(kpfits, (Path, str)):
                kpfits = kpfits[0]
            kpo = KPO(kpfits, input_format="KPFITS")

        model = cls(
            parameters,
            kpo,
            model_type=model_type,
            pos_param=pos_param,
        )
        model.kpfits = kpfits
        return model

    @classmethod
    def from_yaml(cls, path: Path | str, data_file: Path | None = None):
        with open(path) as f:
            mdict = yaml.safe_load(f)
        parameters = parse_parameters(mdict["parameters"])
        args = mdict.get("args", [])
        if isinstance(args, dict):
            args = list(args.values())
        args = [parameters] + args
        kwargs = mdict.get("kwargs", {})
        model = cls.from_kpfits(*args, **kwargs)
        model.yaml_file = path
        return model

    def to_yaml(
        self,
        path: Path | str,
        overwrite: bool = False,
        kpfits: Path | str | list[Path | str] | None = None,
    ):
        path = Path(path)

        model_dict = {}
        model_dict["class"] = self.__class__.__name__
        model_dict["parameters"] = unparse_parameters(self.parameters)
        model_dict["args"] = {}
        if self.kpfits is not None and kpfits is not None:
            raise TypeError(
                "The kpfits argument should be used only when KernelModel.kpfits is not set"
            )
        elif kpfits is None:
            if self.kpfits is None:
                raise ValueError(
                    "kpfits is not set for this model. Pass it as an argument."
                )
            kpfits = self.kpfits
        model_dict["args"]["kpfits"] = kpfits
        kwargs = find_args(self.from_kpfits, argtype="kwargs")
        model_dict["kwargs"] = {}
        for kwarg in kwargs:
            model_dict["kwargs"][kwarg] = getattr(self, kwarg)

        if path.exists() and not overwrite:
            raise FileExistsError(
                f"The file {path} already exists. Use overwrite=True to overwrite it."
            )
        with open(path, mode="w") as f:
            yaml.dump(model_dict, f)

    def _log_likelihood_split(self, p: dict) -> float:
        sigma = np.array([p[f"sigma{i}"] ** 2 for i in range(self.kpo.kp.shape[0])])
        s2 = self.kpo.ekp**2 + sigma[:, None]
        kp_mod = self.forward(p)
        return -0.5 * np.sum(np.log(2 * np.pi * s2) + (kp_mod - self.kpo.kp) ** 2 / s2)

    def _log_likelihood_shared(self, p: dict) -> float:
        s2 = self.kpo.ekp**2 + p["sigma"] ** 2
        kp_mod = self.forward(p)
        return -0.5 * np.sum(np.log(2 * np.pi * s2) + (kp_mod - self.kpo.kp) ** 2 / s2)
