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
    # If conformer present, we keep it; otherwise generate one
    if mol.GetNumConformers() == 0:
        if AllChem.EmbedMolecule(mol, randomSeed=0xBEEF) != 0:
            return None, "3D embedding failed."
        AllChem.UFFOptimizeMolecule(mol)
    return mol, None

def detect_features(mol):
    fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdef)
    return factory.GetFeaturesForMol(mol)

def draw_mol_with_pharmacophore(mol, feats, width=800, height=600):
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
        # optional labels:
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

def first_nonwater_ligand_from_pdb(pdb_text: str):
    """
    Simple heuristic: read HETATM/HETNAM records, ignore water (HOH/WAT), return first ligand code.
    """
    ligand = None
    for line in pdb_text.splitlines():
        if line.startswith(("HET ", "HETNAM")):
            code = line[7:10].strip()
            if code and code.upper() not in {"HOH", "WAT", "DOD"}:
                ligand = code.upper()
                break
    if ligand is None:
        return None, "No non-water ligand detected."
    return ligand, None

def fetch_rcsb_ideal_sdf(lig_code: str):
    # RCSB Chemical Component idealized coordinates
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
st.caption("RDKit + py3Dmol • Works great from VS Code via Streamlit")

tabs = st.tabs(["SMILES input", "PDB → bound ligand"])

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

# --- Tab 2: PDB → Ligand ---
with tabs[1]:
    st.subheader("PDB ID → detect first ligand → Pharmacophore")
    st.write("Note: We fetch the **first non-water ligand** in the PDB file and compute its pharmacophore.")
    pdb_id = st.text_input("Enter a PDB ID (e.g., 1A4K, 6LU7, 4DJH)", "6LU7").strip()
    if st.button("Fetch from PDB", type="primary"):
        if not pdb_id or len(pdb_id) < 4:
            st.error("Please enter a valid 4-character PDB ID.")
        else:
            with st.spinner("Downloading PDB and ligand..."):
                pdb_text, err = fetch_pdb_text(pdb_id)
                if err:
                    st.error(err)
                else:
                    lig, err = first_nonwater_ligand_from_pdb(pdb_text)
                    if err:
                        st.error(err)
                    else:
                        sdf_bytes, err = fetch_rcsb_ideal_sdf(lig)
                        if err:
                            st.warning(f"{err} Trying to proceed via SMILES embedding instead (may differ from crystal).")
                        if sdf_bytes:
                            mol, err = mol_from_sdf_bytes(sdf_bytes)
                            if err:
                                st.error(err)
                                st.stop()
                        else:
                            # Fallback: create a mol from the 3-letter code as a name via PubChem would need internet + PUG;
                            # keeping it simple—tell user to switch to SMILES tab if this fails.
                            st.error("Could not retrieve SDF. If you know the ligand SMILES, use the SMILES tab.")
                            st.stop()

                        feats = detect_features(mol)
                        st.success(f"Detected ligand: **{lig}**")
                        view = draw_mol_with_pharmacophore(mol, feats)
                        st.components.v1.html(view._make_html(), height=650)
