"""Stand-in for the mamba-ssm package (Linux + CUDA only), so rPPG-Toolbox can be
imported without it. Only PhysMamba needs the real thing; selecting it fails loudly."""


class Mamba:
    def __init__(self, *args, **kwargs):
        raise ImportError("PhysMamba needs the real mamba-ssm package (Linux + CUDA only)")
