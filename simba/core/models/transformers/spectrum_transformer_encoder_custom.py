import torch
from depthcharge.transformers import (
    SpectrumTransformerEncoder,
)  # PeptideTransformerEncoder,

from simba.core.data.encoding import OneHotEncoding


class SpectrumTransformerEncoderCustom(SpectrumTransformerEncoder):
    def __init__(
        self,
        *args,
        use_adduct: bool = False,
        categorical_adducts: bool = False,
        adduct_mass_map: str = "",
        use_ce: bool = False,
        use_ion_activation: bool = False,
        use_ion_method: bool = False,
        **kwargs,
    ):
        """
        Custom Spectrum Transformer Encoder with optional metadata usage.

        Attributes
        ----------
        use_adduct: bool
            use adduct info during training
        categorical_adduct: bool
            convert adduct mass to vector
        adduct_mass_map: str
            file that maps adduct masses to vectors
        use_ce: bool
            use collision energy during training
        use_ion_activation: bool
            use ion activation info during training
        use_ion_method: bool
            use ionization method during training
        """
        super().__init__(*args, **kwargs)
        self.use_adduct = use_adduct
        self.categorical_adducts = categorical_adducts
        self.adduct_mass_map = adduct_mass_map
        self.use_ce = use_ce
        self.use_ion_activation = use_ion_activation
        self.use_ion_method = use_ion_method

    def precursor_hook(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        **kwargs: dict,
    ):
        device = mz_array.device
        dtype = mz_array.dtype
        batch_size = mz_array.shape[0]

        placeholder = torch.zeros(
            (batch_size, self.d_model), dtype=dtype, device=device
        )
        precursor_mass = kwargs["precursor_mass"].float().to(device).view(batch_size)
        placeholder[:, 0] = precursor_mass

        precursor_charge = (
            kwargs["precursor_charge"].float().to(device).view(batch_size)
        )
        placeholder[:, 1] = precursor_charge

        current_idx = 2  # keep track of where to insert metadata

        # Initialize metadata encoder if any metadata features are used
        metadata_encoder = None
        if (
            self.use_adduct or self.use_ion_activation or self.use_ion_method
        ) and self.adduct_mass_map:
            metadata_encoder = OneHotEncoding(self.adduct_mass_map)

        if self.use_adduct:
            ionmode = kwargs["ionmode"].float().to(device).view(batch_size)
            placeholder[:, current_idx] = ionmode
            current_idx += 1

            adduct_mass = kwargs["adduct_mass"].float().to(device).view(batch_size)
            placeholder[:, current_idx] = adduct_mass
            current_idx += 1

            if self.categorical_adducts:
                ion_mode_str_list = [
                    ("positive" if ionmode[i].item() > 0 else "negative")
                    for i in range(batch_size)
                ]
                adduct_vectors = []
                for i in range(batch_size):
                    adduct_list = metadata_encoder.encode_adduct(
                        adduct_mass=float(adduct_mass[i].item()),
                        ion_mode=ion_mode_str_list[i],
                    )
                    adduct_vectors.append(adduct_list)
                # Convert list[list] → tensor B × F
                adduct_tensor = torch.tensor(adduct_vectors, dtype=dtype, device=device)
                stop_idx = current_idx + adduct_tensor.shape[1]
                placeholder[:, current_idx:stop_idx] = adduct_tensor
                current_idx = stop_idx

        if self.use_ce:
            ce = kwargs["ce"].float().to(device).view(batch_size)
            placeholder[:, current_idx] = ce
            current_idx += 1

        if self.use_ion_activation:
            ia = kwargs["ion_activation"].float().to(device).view(batch_size)
            ia_encoded = metadata_encoder.encode_ion_activation(ia)
            stop_idx = current_idx + len(ia_encoded)
            placeholder[:, current_idx:stop_idx] = ia
            current_idx = stop_idx

        if self.use_ion_method:
            im = kwargs["ion_method"].float().to(device).view(batch_size)
            im_encoded = metadata_encoder.encode_ionization_method(im)
            stop_idx = current_idx + len(im_encoded)
            placeholder[:, current_idx:stop_idx] = im
            current_idx = stop_idx

        return placeholder
