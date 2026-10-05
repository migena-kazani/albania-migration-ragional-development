# -*- coding: utf-8 -*-
"""
Script 18: Tables 26–28 - Moran's I and Spatial Lag Model
Produces:
  - Table 26. Global Moran's I for Net Migration and Regional GDP (999 permutations)
  - Table 27. Maximum-Likelihood Spatial Lag Model of Net Migration (N=12)
  - Table 28. Model Summary: Spatial Lag Model (N=12)
"""
import json
import numpy as np
from libpysal.weights import W
from esda.moran import Moran
from spreg import ML_Lag
import warnings
warnings.filterwarnings('ignore')

# First-order queen contiguity, verified against INSTAT / administrative boundary sources
NEIGHBORS = {
    'Shkodër': ['Lezhë', 'Kukës'],
    'Kukës': ['Shkodër', 'Lezhë', 'Dibër'],
    'Lezhë': ['Shkodër', 'Kukës', 'Dibër', 'Tiranë', 'Durrës'],
    'Dibër': ['Kukës', 'Lezhë', 'Durrës', 'Tiranë', 'Elbasan'],
    'Durrës': ['Lezhë', 'Dibër', 'Tiranë'],
    'Tiranë': ['Lezhë', 'Dibër', 'Durrës', 'Elbasan'],
    'Elbasan': ['Dibër', 'Tiranë', 'Fier', 'Berat', 'Korçë'],
    'Fier': ['Elbasan', 'Berat', 'Vlorë'],
    'Berat': ['Elbasan', 'Fier', 'Vlorë', 'Gjirokastër', 'Korçë'],
    'Korçë': ['Elbasan', 'Berat', 'Gjirokastër'],
    'Gjirokastër': ['Berat', 'Korçë', 'Vlorë'],
    'Vlorë': ['Fier', 'Berat', 'Gjirokastër'],
}


def main():
    for r, nbrs in NEIGHBORS.items():
        for n in nbrs:
            assert r in NEIGHBORS[n], f"Asymmetry: {r} in {n}'s list but not vice versa"
    print("Adjacency matrix verified symmetric.\n")

    w = W(NEIGHBORS)
    w.transform = 'r'

    cl = json.load(open('../output/clustering.json'))
    regs = cl['regions']
    netmig = np.array(cl['mean_netmig'])
    gdp = np.array(cl['mean_gdp'])
    unemp = np.array(cl['mean_unemp'])

    order = w.id_order
    idx = [regs.index(r) for r in order]
    netmig_o, gdp_o, unemp_o = netmig[idx], gdp[idx], unemp[idx]

    mi_netmig = Moran(netmig_o, w, permutations=999)
    mi_gdp = Moran(gdp_o, w, permutations=999)
    print("Table 26. Global Moran's I (999 permutations)\n")
    print(f"Net migration: I={mi_netmig.I:.3f}, z={mi_netmig.z_sim:.2f}, p={mi_netmig.p_sim:.3f}")
    print(f"Regional GDP:  I={mi_gdp.I:.3f}, z={mi_gdp.z_sim:.2f}, p={mi_gdp.p_sim:.3f}")

    X = np.column_stack([gdp_o, unemp_o])
    y = netmig_o.reshape(-1, 1)
    mllag = ML_Lag(y, X, w=w, name_y='netmig', name_x=['gdp', 'unemp'], name_w='queen_contiguity')
    print("\nTable 27. Maximum-Likelihood Spatial Lag Model of Net Migration (N=12)\n")
    print(mllag.summary)

    print("\nTable 28. Model Summary: Spatial Lag Model")
    print(f"Pseudo R2 = {mllag.pr2:.4f}, Log-likelihood = {mllag.logll:.2f}, "
          f"AIC = {mllag.aic:.2f}, Schwarz = {mllag.schwarz:.2f}")

    json.dump({'moran_netmig_I': mi_netmig.I, 'moran_netmig_p': mi_netmig.p_sim,
               'moran_gdp_I': mi_gdp.I, 'moran_gdp_p': mi_gdp.p_sim,
               'betas': mllag.betas.flatten().tolist(), 'pr2': mllag.pr2,
               'logll': mllag.logll, 'aic': mllag.aic, 'schwarz': mllag.schwarz},
              open('../output/tables22_24_spatial.json', 'w'), indent=2)


if __name__ == "__main__":
    main()
