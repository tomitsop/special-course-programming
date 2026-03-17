######################
###    PACKAGES    ###
######################
import os
import pandas as pd
import numpy as np


#############################
###    LOAD INPUT DATA    ###
#############################

# Set the current working path
current_path = os.getcwd()
data_path = os.path.join(current_path, 'data')

# Import data
df_bus_load = pd.read_csv(os.path.join(data_path, "df_bus_load_added_abroad_final.csv")) # load data for one year
df_bus_load.index = range(1, len(df_bus_load) + 1)
print(df_bus_load.head())
df_bus = pd.read_csv(os.path.join(data_path, "df_bus_final.csv")) # bus ID and zone location
df_branch = pd.read_csv(os.path.join(data_path, "df_branch_final.csv")) # lines and line connections from node to node 
df_plants = pd.read_csv(os.path.join(data_path, "df_gen_final.csv"), sep=";") # generators id and type, bus location, capacity and marginal costs
# Uncomment this line if you want to load the high renewable scenario
# df_plants = pd.read_csv(os.path.join(data_path, "df_gen_final_high_RES.csv"))
incidence = pd.read_csv(os.path.join(data_path, "matrix_A_final.csv")) # Line x Node directed matrix
susceptance = pd.read_csv(os.path.join(data_path, "matrix_Bd_final.csv")) # bus susceptance

# Load Renewable capacity factors
df_pv = pd.read_excel(os.path.join(data_path, "data_renew_2015.xlsx"), sheet_name="pv", header=0) # solar power factors
df_pv.columns = df_pv.columns.astype(str)
print(df_pv.head())
df_wind = pd.read_excel(os.path.join(data_path, "data_renew_2015.xlsx"), sheet_name="onshore", header=0) # onshore wind power factors
df_wind.columns = df_wind.columns.astype(str)
df_wind_off = pd.read_excel(os.path.join(data_path, "data_renew_2015.xlsx"), sheet_name="offshore", header=0) # offshore wind power factors
df_wind_off.columns = df_wind_off.columns.astype(str)


#########################
###    CREATE SETS    ###
#########################

# Time steps (1-based)
T = list(range(1, len(df_bus_load) + 1))

# Renewable types
R = ["PV", "Wind", "Wind Offshore"]

# Conventional plants (non-renewables)
P = df_plants.loc[~df_plants["Type"].isin(R), "GenID"].tolist()

# Buses, lines, zones
N = df_bus["BusID"].tolist()
L = df_branch["BranchID"].tolist()
Z = sorted(df_bus["Zone"].unique())

# Flow-based zone sets
Z_FBMC = Z[:len(Z) - 3]
Z_not_in_FBMC = Z[len(Z) - 3:]

# Nodes in / out of FBMC
N_FBMC = df_bus.loc[df_bus["Zone"].isin(Z_FBMC), "BusID"].tolist()
N_not_in_FBMC = df_bus.loc[~df_bus["Zone"].isin(Z_FBMC), "BusID"].tolist()

print("Printing Sets and lists")
print("T (time steps):", T[:10])
print("R (renewables):", R)
print("P (non-renewable plants):", P[:10])
print("N (bus IDs):", N[:10])
print("L (branch IDs):", L[:10])
print("Z (zones):", Z)
print("Z_FBMC:", Z_FBMC)
print("Z_not_in_FBMC:", Z_not_in_FBMC)
print("N_FBMC (nodes in Z_FBMC):", N_FBMC[:10])
print("N_not_in_FBMC (nodes not in Z_FBMC):", N_not_in_FBMC[:10])


###########################################
###    ZONE ASSIGNMENT FOR PLANTS       ###
###########################################

def replaced_zones():
    zone_p_new = []
    for i in df_plants["OnBus"]:
        matching_zone = df_bus.loc[df_bus["BusID"] == i, "Zone"].values
        if len(matching_zone) > 0:
            zone_p_new.append(matching_zone[0])
        else:
            zone_p_new.append(None)
    return zone_p_new

df_plants["Zone"] = replaced_zones()

# Here I can adjust line capacities to tighten/lighten the grid
df_branch["Pmax"] = 0.6 * df_branch["Pmax"]
# df_branch["Pmax"] = 0.85 * df_branch["Pmax"]
df_bus["ZoneRes"] = df_bus["Zone"]


#########################################
###    REDISPATCHABLE PLANTS (P_RD)   ###
#########################################

# Only Hard Coal and Gas plants are considered as redispatched plants
P_RD = df_plants.loc[
    (df_plants["Type"].isin(["Hard Coal", "Gas/CCGT"])) &
    (df_plants["Zone"].isin(Z_FBMC)),
    "GenID"
].tolist()

print("P_RD (redispatchable plants):", P_RD[:10])


#########################################
###    CREATE MAPPING DICTIONARIES    ###
#########################################

# Nodes in each zone
n_in_z = {
    z: [n for n in N if df_bus.loc[df_bus["BusID"] == n, "Zone"].iloc[0] == z]
    for z in Z
}

# Plants at each node
p_at_n = {
    n: [p for p in P if df_plants.loc[df_plants["GenID"] == p, "OnBus"].iloc[0] == n]
    for n in N
}

# Redispatchable plants at each node
p_rd_at_n = {
    n: [p for p in P_RD if df_plants.loc[df_plants["GenID"] == p, "OnBus"].iloc[0] == n]
    for n in N
}

# Plants in each zone
p_in_z = {
    z: [p for p in P if df_plants.loc[df_plants["GenID"] == p, "Zone"].iloc[0] == z]
    for z in Z
}

# Zone-to-zone connectivity
z_to_z = {
    z: [
        zz for zz in Z
        if zz in df_bus.loc[
            (
                (df_bus["BusID"].isin(df_branch.loc[df_branch["FromBus"].isin(n_in_z[z]), "ToBus"])) |
                (df_bus["BusID"].isin(df_branch.loc[df_branch["ToBus"].isin(n_in_z[z]), "FromBus"]))
            ),
            "Zone"
        ].unique()
        and zz != z
    ]
    for z in Z
}

print("n_in_z:", {k: v[:5] for k, v in n_in_z.items()})
print("p_at_n:", {k: v[:5] for k, v in p_at_n.items()})
print("p_rd_at_n:", {k: v[:5] for k, v in p_rd_at_n.items()})
print("p_in_z:", {k: v[:5] for k, v in p_in_z.items()})
print("z_to_z:", {k: v for k, v in z_to_z.items()})


#########################################
###    INDEX MAPS FOR DIFFERENTIATION ###
#########################################

# conventional plants indexing
P_idx = {p: i for i, p in enumerate(P)}
# Nodes indexing
N_idx = {n: i for i, n in enumerate(N)}
# Lines indexing
L_idx = {l: i for i, l in enumerate(L)}
# Zones indexing
Z_idx = {z: i for i, z in enumerate(Z)}
# FBMC Zones indexing
Z_fb_idx = {z: i for i, z in enumerate(Z_FBMC)}
# Non-FBMC Zones indexing
Z_not_idx = {z: i for i, z in enumerate(Z_not_in_FBMC)}

print(N_idx)

# Mappings in index-space
plants_at_node_idx = {
    n: [P_idx[p] for p in p_at_n[n]] for n in N
}
nodes_in_zone_idx = {
    z: [N_idx[n] for n in n_in_z[z]] for z in Z
}


##############################
###    BASIC FUNCTIONS     ###
##############################

# marginal cost of plants
def get_mc(p):
    return df_plants.loc[df_plants["GenID"] == p, "Costs"].iloc[0]

# maximum marginal cost of redispatchable plants
def find_maximum_mc():
    max_temp = 0.0
    for p in P_RD:
        mc_temp = get_mc(p)
        if mc_temp > max_temp:
            max_temp = mc_temp
    return max_temp

def get_dem(t, n):
    # df_bus_load index is assumed compatible with T (1-based)
    return df_bus_load.loc[t, str(n)]


#########################################
###    RENEWABLE RESOURCE TABLE       ###
#########################################

def create_res_table():
    res_temp = np.zeros((len(T), len(N), len(R)), dtype=float)
    for n in N:
        n_idx = N_idx[n]
        for r in R:
            r_idx = R.index(r)
            zone_temp = df_bus.loc[df_bus["BusID"] == n, "ZoneRes"].iloc[0]
            cap_temp = df_plants.loc[
                (df_plants["Type"] == r) & (df_plants["OnBus"] == n),
                "Pmax"
            ].sum()
            if r == "PV":
                share_temp = df_pv[zone_temp]
            elif r == "Wind":
                share_temp = df_wind[zone_temp]
            else:
                share_temp = df_wind_off[zone_temp]
            res_temp[:, n_idx, r_idx] = 1.5 * cap_temp * share_temp.values
    return res_temp

res_table = create_res_table()

def get_renew(t, n):
    t_idx = T.index(t)
    n_idx = N_idx[n]
    return float(res_table[t_idx, n_idx, :].sum())

def get_renew_zone(t, z):
    nodes = n_in_z[z]
    return sum(get_renew(t, n) for n in nodes)

# Vectorized versions (for use as parameters to differentiable layers)
def get_dem_vec(t):
    """Return demand vector (|N|,) for time t in the correct node order N."""
    return np.array([get_dem(t, n) for n in N], dtype=float)

def get_renew_vec(t):
    """Return renewable availability vector (|N|,) for time t in the correct node order N."""
    t_idx = T.index(t)
    # Sum across renewable types axis
    return res_table[t_idx, :, :].sum(axis=1)


#########################################
###    GENERATION / LINE CAPACITY     ###
#########################################

def get_gen_up(p):
    return df_plants.loc[df_plants["GenID"] == p, "Pmax"].iloc[0]

def get_line_cap(l):
    return df_branch.loc[df_branch["BranchID"] == l, "Pmax"].iloc[0]

# Vectorized arrays for use in cvxpy
cost_gen = np.array([get_mc(p) for p in P], dtype=float)
gmax = np.array([get_gen_up(p) for p in P], dtype=float)
line_cap = np.array([get_line_cap(l) for l in L], dtype=float)


#########################################
###    CROSS-BORDER LINES            ###
#########################################

# Identify which are the cross/border lines by picking from and to bus and checking if they belong in fbmc but in different zones
def find_cross_border_lines():
    cb_lines_temp = []
    for l in L:
        from_bus = df_branch.loc[df_branch["BranchID"] == l, "FromBus"].iloc[0]
        to_bus = df_branch.loc[df_branch["BranchID"] == l, "ToBus"].iloc[0]
        from_zone_temp = df_bus.loc[df_bus["BusID"] == from_bus, "Zone"].iloc[0]
        to_zone_temp = df_bus.loc[df_bus["BusID"] == to_bus, "Zone"].iloc[0]
        if from_zone_temp in Z_FBMC and to_zone_temp in Z_FBMC and from_zone_temp != to_zone_temp:
            cb_lines_temp.append(l)
    return cb_lines_temp

cross_border_lines = find_cross_border_lines()
print("Cross-border lines:", cross_border_lines)


#########################################
###    SUSCEPTANCE / PTDF MATRICES   ###
#########################################

# Base value and slack node
MWBase = 380 ** 2
slack_node = 50
slack_position = N.index(slack_node)

# Raw susceptance matrices
line_sus_mat = np.matmul(susceptance.values / MWBase, incidence.values)
node_sus_mat = np.matmul(
    np.matmul(incidence.values.T, susceptance.values / MWBase),
    incidence.values
)

# Dictionaries (if ever needed)
H_mat = {(l, n): line_sus_mat[L_idx[l], N_idx[n]] for l in L for n in N}
B_mat = {(n, nn): node_sus_mat[N_idx[n], N_idx[nn]] for n in N for nn in N}

# Full B and H matrices (for DC-OPF and DC-flow usage)
B_matrix = node_sus_mat.copy()   # shape (|N|, |N|)
H_matrix = line_sus_mat.copy()   # shape (|L|, |N|)

# Reduced PTDF: remove slack, invert B_, then reinsert slack
line_sus_mat_ = np.delete(line_sus_mat, slack_position, axis=1)
node_sus_mat_ = np.delete(np.delete(node_sus_mat, slack_position, axis=0), slack_position, axis=1)

PTDF_full = np.matmul(line_sus_mat_, np.linalg.inv(node_sus_mat_))
zero_column = np.zeros((len(L), 1))
PTDF_full = np.hstack((PTDF_full[:, :slack_position], zero_column, PTDF_full[:, slack_position:]))

# PTDF restricted to FBMC nodes only
N_FBMC_indices = [N_idx[n] for n in N_FBMC]
PTDF_FBMC = PTDF_full[:, N_FBMC_indices]



#################################
###    CREATE GSK MATRICES    ###
#################################

# Local index for N_FBMC nodes
N_FBMC_idx = {n: i for i, n in enumerate(N_FBMC)}
Z_FBMC_idx = {z: i for i, z in enumerate(Z_FBMC)}

def get_gsk_flat():
    """
    Flat GSK: each node in a zone gets 1 / (#nodes_in_zone) for that zone.
    Shape: (|N_FBMC|, |Z_FBMC|)
    """
    gsk_temp = np.zeros((len(N_FBMC), len(Z_FBMC)), dtype=float)

    for n in N_FBMC:
        zone_temp = df_bus.loc[df_bus["BusID"] == n, "Zone"].iloc[0]
        n_index = N_FBMC_idx[n]
        z_index = Z_FBMC_idx[zone_temp]

        nodes_in_zone = df_bus[df_bus["Zone"] == zone_temp]["BusID"].tolist()
        gsk_value_temp = 1.0 / len(nodes_in_zone)
        gsk_temp[n_index, z_index] = gsk_value_temp

    return gsk_temp

gsk_flat = get_gsk_flat()


def get_gsk_flat_unit():
    """
    Flat unit GSK: equal weight for conventional nodes in each zone.
    Only nodes with conventional generation get a non-zero GSK.
    Shape: (|N_FBMC|, |Z_FBMC|)
    """
    gsk_temp = np.zeros((len(N_FBMC), len(Z_FBMC)), dtype=float)

    for n in N_FBMC:
        zone_temp = df_bus.loc[df_bus["BusID"] == n, "Zone"].iloc[0]
        n_index = N_FBMC_idx[n]
        z_index = Z_FBMC_idx[zone_temp]

        conv_nodes_in_zone = df_plants.loc[
            (df_plants["Zone"] == zone_temp) & (df_plants["GenID"].isin(P)),
            "OnBus"
        ].unique()

        if n in conv_nodes_in_zone:
            gsk_value_temp = 1.0 / len(conv_nodes_in_zone)
            gsk_temp[n_index, z_index] = gsk_value_temp

    return gsk_temp

gsk_flat_unit = get_gsk_flat_unit()


def get_gsk_pmax():
    """
    Pmax-based GSK: weight at node proportional to installed conventional Pmax
    in that zone.
    Shape: (|N_FBMC|, |Z_FBMC|)
    """
    gsk_temp = np.zeros((len(N_FBMC), len(Z_FBMC)), dtype=float)

    for n in N_FBMC:
        zone_temp = df_bus.loc[df_bus["BusID"] == n, "Zone"].iloc[0]
        n_index = N_FBMC_idx[n]
        z_index = Z_FBMC_idx[zone_temp]

        conv_nodes_in_zone = df_plants.loc[
            (df_plants["Zone"] == zone_temp) & (df_plants["GenID"].isin(P)),
            "OnBus"
        ].unique()

        if n in conv_nodes_in_zone:
            conv_pmax_in_zone = df_plants.loc[
                (df_plants["OnBus"].isin(conv_nodes_in_zone)) & (df_plants["GenID"].isin(P)),
                "Pmax"
            ].sum()

            conv_pmax_at_node = df_plants.loc[
                (df_plants["OnBus"] == n) & (df_plants["GenID"].isin(P)),
                "Pmax"
            ].sum()

            if conv_pmax_in_zone > 0:
                gsk_value_temp = conv_pmax_at_node / conv_pmax_in_zone
                gsk_temp[n_index, z_index] = gsk_value_temp

    return gsk_temp

gsk_pmax = get_gsk_pmax()


def get_gsk_pmax_sub():
    """
    Pmax-based GSK on a subset of generators (Hard Coal, Gas/CCGT in FB zones).
    Used for CNE selection.
    Shape: (|N_FBMC|, |Z_FBMC|)
    """
    P_sub = df_plants.loc[
        (df_plants["Type"].isin(["Hard Coal", "Gas/CCGT"])) &
        (df_plants["Zone"].isin(Z_FBMC)),
        "GenID"
    ].tolist()

    gsk_temp = np.zeros((len(N_FBMC), len(Z_FBMC)), dtype=float)

    for n in N_FBMC:
        zone_temp = df_bus.loc[df_bus["BusID"] == n, "Zone"].iloc[0]
        n_index = N_FBMC_idx[n]
        z_index = Z_FBMC_idx[zone_temp]

        conv_nodes_in_zone = df_plants.loc[
            (df_plants["Zone"] == zone_temp) & (df_plants["GenID"].isin(P_sub)),
            "OnBus"
        ].unique()

        if n in conv_nodes_in_zone:
            conv_pmax_in_zone = df_plants.loc[
                (df_plants["OnBus"].isin(conv_nodes_in_zone)) & (df_plants["GenID"].isin(P_sub)),
                "Pmax"
            ].sum()

            conv_pmax_at_node = df_plants.loc[
                (df_plants["OnBus"] == n) & (df_plants["GenID"].isin(P_sub)),
                "Pmax"
            ].sum()

            if conv_pmax_in_zone > 0:
                gsk_value_temp = conv_pmax_at_node / conv_pmax_in_zone
                gsk_temp[n_index, z_index] = gsk_value_temp

    return gsk_temp

gsk_pmax_sub = get_gsk_pmax_sub()

print("All GSKs built, with column sums:")
print("1) GSK flat:",       np.round(np.sum(gsk_flat, axis=0), 2))
print("2) GSK flat unit:",  np.round(np.sum(gsk_flat_unit, axis=0), 2))
print("3) GSK pmax:",       np.round(np.sum(gsk_pmax, axis=0), 2))
print("4) GSK pmax sub:",   np.round(np.sum(gsk_pmax_sub, axis=0), 2))


###########################
###    CNE SELECTION    ###
###########################

cne_alpha = 0.05 # CNEC selection threshold
gsk_cne = gsk_pmax_sub # GSK used for CNE selection
gsk_mc = gsk_pmax_sub 
frm = 0.05 # flow reliability margin to reduce from total line capacity
include_cb_lines = True

# PTDF_FBMC: (|L|, |N_FBMC|)
PTDF_Z = PTDF_FBMC @ gsk_cne    # (|L|, |Z_FBMC|)

n_z_fbmc = len(Z_FBMC)
n_pairs = int(n_z_fbmc * (n_z_fbmc - 1) / 2)
z2z_temp = np.zeros((len(L), n_pairs))

counter = 0
for i_z in range(n_z_fbmc - 1):
    for j_z in range(i_z + 1, n_z_fbmc):
        z2z_temp[:, counter] = PTDF_Z[:, i_z] - PTDF_Z[:, j_z]
        counter += 1

z2z_temp_abs = np.abs(z2z_temp)
maximum_abs_z2z = np.max(z2z_temp_abs, axis=1)

CNEC = [L[i] for i, x in enumerate(maximum_abs_z2z) if x >= cne_alpha] # CNEC significant if z2z greather than or equal to CNEC threshold

# Add cross-border lines in CNEC list
if include_cb_lines:
    cb_lines = find_cross_border_lines()
    for line in cb_lines:
        if line not in CNEC:
            CNEC.append(line)
    CNEC = list(set(CNEC))

CNEC = [l for l in L if l in CNEC]

CNEC_indices = [L_idx[l] for l in CNEC]

# Nodal and zonal PTDF matrices on CNECs
PTDF_CNEC = PTDF_FBMC[CNEC_indices, :]   # (|CNEC|, |N_FBMC|)
PTDF_Z_CNEC = PTDF_Z[CNEC_indices, :]    # (|CNEC|, |Z_FBMC|)

print(f"CNE selection: {len(CNEC)} CNEs selected at alpha={cne_alpha}")
print("Critical Network Elements and Contingencies (CNEC):", CNEC)