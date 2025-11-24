
import pandas as pd

class AdductHandling:

    def __init__(self, adduct_file_path):
        self.df_adduct = self.load_adduct_info(adduct_file_path)


    def load_adduct_info(self, path_adduct_csv):
        return pd.read_csv(path_adduct_csv, delimiter=';')

    ### obtain information for adduct handling based on adduct mass
    def get_categorical_adduct(self, adduct_mass, ion_mode, ):
        """
        adduct_mass : float
            Target adduct mass (mass shift).
        ion_mode : str
            "positive" or "negative".
        df_adduct : pd.DataFrame
            DataFrame with columns:
            - 'Ion mode'
            - 'Mass' (adduct mass shift)
            - adduct_* columns (adduct_M, adduct_H, adduct_Na, ...)

        Returns
        -------
        list_adduct_elements : list[float or int]
            Values of all adduct_* columns (including adduct_M) for the closest adduct.
            The order is the column order in df_adduct.
        """

        # Keep only rows with the requested ion mode
        mask_mode = self.df_adduct["Ion mode"].str.lower() == ion_mode.lower()
        df_mode = self.df_adduct[mask_mode]

        if df_mode.empty:
            raise ValueError(f"No adducts found for ion mode: {ion_mode}")

        # Find the row whose 'Mass' is closest to the target adduct_mass
        idx_closest = (df_mode["Mass"] - adduct_mass).abs().idxmin()
        best_row = df_mode.loc[idx_closest]

        # All columns that encode adduct composition (including adduct_M)
        adduct_cols = [c for c in self.df_adduct.columns if c.startswith("adduct_")]

        # Extract their values as a list (you can cast to int if you prefer)
        list_adduct_elements = best_row[adduct_cols].tolist()

        return list_adduct_elements