import warnings
warnings.filterwarnings(
    "ignore",
    message="Numerical value without unit or explicit format passed to TimeDelta, assuming days",
    module="astropy"
)
warnings.filterwarnings(
    "ignore",
    message="Numerical value without unit or explicit format passed to TimeDelta, assuming days",
    module="astropy"
)

warnings.filterwarnings(
    "ignore",
    message=".*Unit 'BJD' not supported by the FITS standard.*",
    module="astropy"
)

warnings.filterwarnings(
    "ignore",
    message=".*Unit 'e' not supported by the FITS standard.*",
    module="astropy"
)

warnings.filterwarnings(
    "ignore",
    message=".*'pixels' did not parse as fits unit*",
    module="astropy"
)

warnings.filterwarnings(
    "ignore",
    message=".*did not parse as fits unit*",
    module="astropy"
)

warnings.filterwarnings(
    "ignore",
    message=".*missing from current font*",
)

warnings.filterwarnings(
    "ignore",
    message=".*'partition' will ignore*",
)

warnings.filterwarnings(
    "ignore",
    message=".*DejaVu Sans*",
)
warnings.filterwarnings(
    "ignore",
    message=".*tpfmodel submodule is not available without oktopus installed, which requires a current version of autograd*",
)

warnings.filterwarnings(
    "ignore",
    message=".*Liberation Sans*",
)

warnings.filterwarnings(
    "ignore",
    message=".*force_all_finite*",
)

warnings.filterwarnings(
    "ignore",
    module="seaborn"
)