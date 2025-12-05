from simba.chem_utils import ADDUCT_TO_MASS

ION_ACTIVATION = ["HCD", "CID"]
IONIZATION_METHODS = ["NSI", "ESI", "APCI"]


def encode_adduct(adduct: str):
    adducts = ADDUCT_TO_MASS.keys()
    return [1 if adduct == a else 0 for a in adducts]


def encode_ion_activation(ion_activation: str):
    return [1 if ion_activation == ia else 0 for ia in ION_ACTIVATION]


def encode_ionization_method(ionization_method: str):
    return [1 if ionization_method == im else 0 for im in IONIZATION_METHODS]
