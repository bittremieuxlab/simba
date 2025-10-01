import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

def plot_bars(query_indexes, variable_0, variable_1, 
                            figsize=(20,5), bar_width=0.4, 
                            variable_0_color='b', variable_1_color='r', name_variable_0 ='variable_0', name_variable_1='variable_1',
                            ylim=(0, 1.2), title="variable_0 and variable_1 per Query"):
    """
    Plots variable_0 and variable_1 side-by-side bars for each query.

    Parameters
    ----------
    query_indexes : list or array-like
        Identifiers (x-axis) for each query.
    variable_0 : list or array-like
        variable_0 values for each query.
    variable_1 : list or array-like
        variable_1 values for each query.
    figsize : tuple, optional
        Figure size, default (20,5).
    bar_width : float, optional
        Width of the bars, default 0.4.
    variable_0_color : str, optional
        Color for variable_0 bars, default 'b'.
    variable_1_color : str, optional
        Color for variable_1 bars, default 'r'.
    ylim : tuple, optional
        Y-axis limits for both axes, default (0, 1.2).
    title : str, optional
        Plot title.
    """
    
    query_indexes = np.array(query_indexes)

    fig, ax1 = plt.subplots(figsize=figsize)

    # Left y-axis (variable_0)
    ax1.bar(query_indexes - bar_width/2, variable_0, 
            width=bar_width, color=variable_0_color, alpha=0.6, label=name_variable_0)
    ax1.set_xlabel('Query')
    ax1.set_ylabel(name_variable_0, color=variable_0_color)
    ax1.set_xticks(query_indexes)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=variable_0_color)
    ax1.set_ylim(*ylim)

    # Right y-axis (variable_1)
    ax2 = ax1.twinx()
    ax2.bar(query_indexes + bar_width/2, variable_1, 
            width=bar_width, color=variable_1_color, alpha=0.6, label=name_variable_1)
    ax2.set_ylabel(name_variable_1, color=variable_1_color)
    ax2.tick_params(axis='y', labelcolor=variable_1_color)
    ax2.set_ylim(*ylim)

    # Legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title(title)
    plt.tight_layout()
    plt.show()



from rdkit.Chem import Draw
from rdkit import Chem

def plot_mols(smiles):
    # Convert SMILES to RDKit Mol objects
    for sanitize in [False, True]:
        try:
            mols=[]
            for s in smiles:
                mol= Chem.MolFromSmiles(s, sanitize=sanitize)
                if mol:
                    mols.append(mol)

            # Show them in a grid (Jupyter/IPython will display this directly)

            img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(800,800))
            display(img)  # if in Jupyter, just `img`
            break
        except:
            print(f'Plotting with sanitize={sanitize} did not work. Lets try with the opposite')
       
   