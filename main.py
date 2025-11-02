#   streamlit run main.py

import os
import io
import requests
import streamlit as st
import py3Dmol

from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalFeatures
from rdkit import RDConfig

# ---------------------------
# Helpers
# ---------------------------
def mol_from_smiles_to_3d(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Could not parse SMILES."
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE) != 0:
        return None, "3D embedding failed."
    AllChem.UFFOptimizeMolecule(mol)
    return mol, None

def mol_from_sdf_bytes(data: bytes):
    suppl = Chem.ForwardSDMolSupplier(io.BytesIO(data))
    mols = [m for m in suppl if m is not None]
    if not mols:
        return None, "No molecule found in SDF."
    mol = Chem.AddHs(mols[0], addCoords=True)
    if mol.GetNumConformers() == 0:
        if AllChem.EmbedMolecule(mol, randomSeed=0xBEEF) != 0:
            return None, "3D embedding failed."
        AllChem.UFFOptimizeMolecule(mol)
    return mol, None

def detect_features(mol):
    fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdef)
    return factory.GetFeaturesForMol(mol)

def draw_mol_with_pharmacophore(mol, feats, width=800, height=400):
    mb = Chem.MolToMolBlock(mol)
    view = py3Dmol.view(width=width, height=height)
    view.addModel(mb, "sdf")
    view.setStyle({"stick": {}})
    view.zoomTo()

    colors = {
        "Donor": "blue",
        "Acceptor": "red",
        "Aromatic": "purple",
        "Hydrophobe": "orange",
        "LumpedHydrophobe": "orange",
        "PosIonizable": "green",
        "NegIonizable": "yellow",
        "ZnBinder": "gray",
    }
    default_radius = 0.9

    for f in feats:
        fam = f.GetFamily()
        x, y, z = f.GetPos()
        view.addSphere({
            "center": {"x": float(x), "y": float(y), "z": float(z)},
            "radius": default_radius,
            "color": colors.get(fam, "white"),
            "alpha": 0.5
        })
        view.addLabel(fam, {
            "position": {"x": float(x), "y": float(y), "z": float(z)},
            "backgroundOpacity": 0.0,
            "fontSize": 10
        })
    return view

def fetch_pdb_text(pdb_id: str):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return None, f"Could not download PDB file for {pdb_id}."
    return r.text, None

def list_ligands_in_pdb(pdb_text: str):
    """
    Return a dict: {LIGCODE: {"instances": [("CHAIN","RESSEQ","ICODE")...], "count": int}}
    Uses HETATM lines; ignores water-like residues.
    """
    ignore = {"HOH", "WAT", "DOD"}
    ligmap = {}
    for line in pdb_text.splitlines():
        if not line.startswith("HETATM"):
            continue
        resname = line[17:20].strip().upper()
        if resname in ignore:
            continue
        chain = line[21].strip() or "-"
        resseq = line[22:26].strip()
        icode = line[26].strip() or ""
        key = (resname, chain, resseq, icode)
        if resname not in ligmap:
            ligmap[resname] = {"_set": set(), "instances": []}
        if key not in ligmap[resname]["_set"]:
            ligmap[resname]["_set"].add(key)
            ligmap[resname]["instances"].append((chain, resseq, icode))
    # finalize counts
    output = {}
    for code, d in ligmap.items():
        output[code] = {
            "instances": d["instances"],
            "count": len(d["instances"])
        }
    return output

def fetch_rcsb_ideal_sdf(lig_code: str):
    url = f"https://files.rcsb.org/ligands/download/{lig_code}_ideal.sdf"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return None, f"Could not fetch ideal SDF for ligand {lig_code}."
    return r.content, None

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="3D Pharmacophore Viewer", layout="centered")
st.title("🔬 3D Pharmacophore Viewer")
st.caption("RDKit + py3Dmol")

tabs = st.tabs(["SMILES input", "PDB → choose ligand"])

# --- Tab 1: SMILES ---
with tabs[0]:
    st.subheader("SMILES → 3D → Pharmacophore")
    smiles = st.text_input("Enter a small-molecule SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")  # aspirin default
    if st.button("Render from SMILES", type="primary"):
        with st.spinner("Generating 3D and features..."):
            mol, err = mol_from_smiles_to_3d(smiles)
            if err:
                st.error(err)
            else:
                feats = detect_features(mol)
                view = draw_mol_with_pharmacophore(mol, feats)
                st.components.v1.html(view._make_html(), height=650)

# --- Tab 2: PDB → choose ligand ---
with tabs[1]:
    st.subheader("PDB ID → list ligands → Pharmacophore")
    st.write("We’ll list all **non-water ligands** detected in the PDB (by 3-letter code). Pick one to view its pharmacophore.")

    pdb_id = st.text_input("Enter a PDB ID (e.g., 1A4K, 6LU7, 4DJH)", "6LU7").strip()
    if st.button("Fetch PDB", type="primary"):
        if not pdb_id or len(pdb_id) < 4:
            st.error("Please enter a valid 4-character PDB ID.")
        else:
            with st.spinner("Downloading PDB and scanning ligands..."):
                pdb_text, err = fetch_pdb_text(pdb_id)
                if err:
                    st.error(err)
                else:
                    ligs = list_ligands_in_pdb(pdb_text)
                    st.session_state.pdb_text = pdb_text
                    st.session_state.ligands = ligs
                    if not ligs:
                        st.warning("No non-water ligands found.")
                    else:
                        st.success(f"Found {len(ligs)} ligand code(s).")

    # Ensure previous results persist
    ligs = st.session_state.get("ligands", {})
    if "lig_choice" not in st.session_state:
        st.session_state.lig_choice = None

    if ligs:
        def label_for(code, info):
            inst = info["instances"]
            pretty = ", ".join([f"{c}/{r}{i}" if i else f"{c}/{r}" for c, r, i in inst[:6]])
            if len(inst) > 6:
                pretty += ", …"
            return f"{code} ({info['count']} instance{'s' if info['count']!=1 else ''}: {pretty})"

        options = sorted(ligs.keys())
        display_map = {label_for(c, ligs[c]): c for c in options}

        choice_label = st.selectbox(
            "Choose a ligand code",
            list(display_map.keys()),
            index=0 if st.session_state.lig_choice is None else list(display_map.keys()).index(
                [k for k, v in display_map.items() if v == st.session_state.lig_choice][0]
            ) if st.session_state.lig_choice in display_map.values() else 0,
        )
        st.session_state.lig_choice = display_map[choice_label]

        if st.button("Render selected ligand", type="primary"):
            lig = st.session_state.lig_choice
            with st.spinner(f"Fetching ideal SDF for {lig} and computing pharmacophore..."):
                sdf_bytes, err = fetch_rcsb_ideal_sdf(lig)
                if err:
                    st.error(err)
                else:
                    mol, err = mol_from_sdf_bytes(sdf_bytes)
                    if err:
                        st.error(err)
                    else:
                        feats = detect_features(mol)
                        st.success(f"Rendering ligand **{lig}**")
                        view = draw_mol_with_pharmacophore(mol, feats)
                        st.components.v1.html(view._make_html(), height=650)

# Footer note
st.caption("Tip: Ideal SDF shows the canonical ligand; crystallographic pose can differ.")
