import torch

from depthcharge.transformers import (
    SpectrumTransformerEncoder,
    # PeptideTransformerEncoder,
)


class SpectrumTransformerEncoderCustom(SpectrumTransformerEncoder):
    def __init__(self, *args, use_extra_metadata: bool = False, **kwargs):
        """
        Custom Spectrum Transformer Encoder with optional precursor metadata usage.
        
        Parameters
        ----------
        use_extra_metadata : bool, optional
            Whether to include extra precursor metadata in the encoding (default: False).
        """
        super().__init__(*args, **kwargs)
        self.use_extra_metadata = use_extra_metadata

    def precursor_hook(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        **kwargs: dict,
    ):
        if self.use_extra_metadata:
            ####do something
            print('Using extra metadata')
            mass_precursor = torch.tensor(kwargs["precursor_mass"].float())
            charge_precursor = torch.tensor(kwargs["precursor_charge"].float())
            ionization_mode_precursor= torch.tensor(kwargs["ionization_mode"].float())
            adduct_mass_precursor= torch.tensor(kwargs["adduct_mass"].float())


            # placeholder
            placeholder = torch.zeros((mz_array.shape[0], self.d_model)).type_as(mz_array)

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
