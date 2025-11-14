import torch
from simba.adduct_handling.adduct_handling import AdductHandling

from depthcharge.transformers import (
    SpectrumTransformerEncoder,
    # PeptideTransformerEncoder,
)


class SpectrumTransformerEncoderCustom(SpectrumTransformerEncoder):
    def __init__(self, *args, use_extra_metadata: bool = False, use_categorical_adducts: bool= False, 
                      adduct_info_csv:str="",  **kwargs):
        """
        Custom Spectrum Transformer Encoder with optional precursor metadata usage.
        
        Parameters
        ----------
        use_extra_metadata : bool, optional
            Whether to include extra precursor metadata in the encoding (default: False).
        """
        super().__init__(*args, **kwargs)
        self.use_extra_metadata = use_extra_metadata
        self.use_categorical_adducts=use_categorical_adducts
        self.adduct_info_csv =adduct_info_csv

    def precursor_hook(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        **kwargs: dict,
    ):
    # scalar / per-batch metadata
        if self.use_extra_metadata:
            device = mz_array.device
            dtype  = mz_array.dtype
            B = mz_array.shape[0]   # batch size

            # scalar metadata as tensors
            mass_precursor = kwargs["precursor_mass"].float().to(device)
            charge_precursor = kwargs["precursor_charge"].float().to(device)
            ionization_mode_precursor = kwargs["ionmode"].float().to(device)
            adduct_mass_precursor = kwargs["adduct_mass"].float().to(device)

            # placeholder
            placeholder = torch.zeros((B, self.d_model), dtype=dtype, device=device)

            # ---- FIRST 4 FIXED FEATURES ----
            base_meta = torch.cat(
                (
                    mass_precursor.view(B, 1),
                    charge_precursor.view(B, 1),
                    ionization_mode_precursor.view(B, 1),
                    adduct_mass_precursor.view(B, 1),
                ),
                dim=-1,
            )
            placeholder[:, 0:4] = base_meta

            # ======================================================
            #       FIX: CONVERT ionization_mode_precursor → STRINGS
            # ======================================================
            # ionization_mode_precursor is numeric (e.g., +1 / -1)
            ion_mode_str_list = [
                "positive" if ionization_mode_precursor[i].item() > 0 else "negative"
                for i in range(B)
            ]
            # Now we have: ["positive", "negative", ...] for the batch

            # ===============================================
            #          USE CATEGORICAL ADDUCTS
            # ===============================================
            if self.use_categorical_adducts:
                adduct_obj = AdductHandling(self.adduct_info_csv)
                # For each spectrum in the batch compute adduct vector
                adduct_vectors = []
                for i in range(B):
                    adduct_list = adduct_obj.get_categorical_adduct(
                        adduct_mass=float(adduct_mass_precursor[i].item()),
                        ion_mode=ion_mode_str_list[i],
                    )
                    adduct_vectors.append(adduct_list)

                # Convert list[list] → tensor B × F
                adduct_tensor = torch.tensor(adduct_vectors, dtype=dtype, device=device)

                # Write categorical adducts after the first 4 positions
                placeholder[:, 4:4 + adduct_tensor.shape[1]] = adduct_tensor
        else:
            mass_precursor = torch.tensor(kwargs["precursor_mass"].float())
            charge_precursor = torch.tensor(kwargs["precursor_charge"].float())

            # placeholder
            placeholder = torch.zeros((mz_array.shape[0], self.d_model)).type_as(mz_array)

            placeholder[:, 0:2] = torch.cat(
                (mass_precursor.view(-1, 1), charge_precursor.view(-1, 1)), dim=-1
            )
        return placeholder
