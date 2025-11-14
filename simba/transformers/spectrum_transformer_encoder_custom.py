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
        if self.use_extra_metadata:
            ####do something
            mass_precursor = torch.tensor(kwargs["precursor_mass"].float())
            charge_precursor = torch.tensor(kwargs["precursor_charge"].float())
            ionization_mode_precursor= torch.tensor(kwargs["ionmode"].float())
            adduct_mass_precursor= torch.tensor(kwargs["adduct_mass"].float())


            # placeholder


            placeholder = torch.zeros((mz_array.shape[0], self.d_model)).type_as(mz_array)

            if self.use_categorical_adducts:
                ## USING categorical variables
                ion_mode_string = 'positive' if ionization_mode_precursor == 1 else 'negative'
                adduct_obj = AdductHandling(self.adduct_info_csv)
                adduct_elements_list = adduct_obj.get_categorical_adduct(
                    adduct_mass_precursor, ion_mode_string
                )

                # ---- default 4 metadata features in the first positions ----
                placeholder[:, 0:4] = torch.cat(
                    (
                        mass_precursor.view(-1, 1),
                        charge_precursor.view(-1, 1),
                        ionization_mode_precursor.view(-1, 1),
                        adduct_mass_precursor.view(-1, 1),
                    ),
                    dim=-1,
                )

                # ---- append adduct element features after the 4 defaults ----
                # adduct_elements_list is e.g. [adduct_M, adduct_H, adduct_Na, ...]
                adduct_elements_tensor = torch.tensor(
                    adduct_elements_list,
                    dtype=mz_array.dtype,
                    device=mz_array.device,
                ).view(1, -1)  # shape (1, n_adduct_features)

                # broadcast to batch size
                adduct_elements_tensor = adduct_elements_tensor.repeat(
                    mz_array.shape[0], 1
                )  # (batch_size, n_adduct_features)

                # put them after the 4 scalar metadata features
                start_idx = 4
                end_idx = 4 + adduct_elements_tensor.shape[1]
                placeholder[:, start_idx:end_idx] = adduct_elements_tensor
            else:
                placeholder[:, 0:4] = torch.cat(
                    (mass_precursor.view(-1, 1), 
                    charge_precursor.view(-1, 1),
                    ionization_mode_precursor.view(-1,1),
                    adduct_mass_precursor.view(-1,1),
                    ), 
                    dim=-1
                )

        else:
            mass_precursor = torch.tensor(kwargs["precursor_mass"].float())
            charge_precursor = torch.tensor(kwargs["precursor_charge"].float())

            # placeholder
            placeholder = torch.zeros((mz_array.shape[0], self.d_model)).type_as(mz_array)

            placeholder[:, 0:2] = torch.cat(
                (mass_precursor.view(-1, 1), charge_precursor.view(-1, 1)), dim=-1
            )
        return placeholder
