import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs, Draw
from rdkit import RDLogger
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import StandardScaler
import io
import base64
from PIL import Image

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Set page config with custom theme
st.set_page_config(
    page_title="BBB Permeability Predictor", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prediction-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .permeable-card {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
    }
    
    .non-permeable-card {
        background: linear-gradient(135deg, #f44336, #da190b);
        color: white;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stProgress .st-bo {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

class MolecularFeatureGenerator:
    """Enhanced molecular feature extraction with progress tracking."""
    def __init__(self, fp_size=2048):
        self.fp_size = fp_size
        self.descriptor_names = [
            'MW', 'LogP', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
            'NumAromaticRings', 'TPSA', 'NumSaturatedRings', 'NumAliphaticRings',
            'RingCount', 'NumHeteroatoms', 'NumSaturatedHeterocycles',
            'NumAromaticHeterocycles', 'BertzCT', 'LabuteASA', 'BalabanJ', 'Kappa1',
            'Kappa2', 'Kappa3', 'MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex',
            'MinAbsEStateIndex', 'NumLipinskiHBD', 'NumLipinskiHBA','fr_NH0', 'fr_NH1', 
            'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_C_O', 
            'fr_C_O_noCOO', 'fr_benzene', 'fr_phenol', 'VSA_EState1', 'VSA_EState2', 
            'VSA_EState3', 'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SMR_VSA1', 
            'SMR_VSA2', 'SMR_VSA3', 'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3'
        ]
        self.feature_names = self.descriptor_names + [f'morgan_bit_{j}' for j in range(self.fp_size)]
        self.failed_molecules = []

    def smiles_to_mol(self, smiles):
        try:
            mol = Chem.MolFromSmiles(str(smiles).strip())
            if mol is None: return None
            mol = Chem.AddHs(mol)
            return mol
        except Exception:
            return None

    def calculate_molecular_descriptors(self, mol):
        if mol is None: return None
        try:
            mol_no_h = Chem.RemoveHs(mol)
            desc_dict = {
                'MW': Descriptors.MolWt(mol), 'LogP': Descriptors.MolLogP(mol),
                'NumHDonors': Descriptors.NumHDonors(mol), 'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol), 'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                'TPSA': Descriptors.TPSA(mol), 'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
                'NumAliphaticRings': Descriptors.NumAliphaticRings(mol), 'RingCount': Descriptors.RingCount(mol),
                'NumHeteroatoms': Descriptors.NumHeteroatoms(mol), 'NumSaturatedHeterocycles': Descriptors.NumSaturatedHeterocycles(mol),
                'NumAromaticHeterocycles': Descriptors.NumAromaticHeterocycles(mol), 'BertzCT': Descriptors.BertzCT(mol),
                'LabuteASA': Descriptors.LabuteASA(mol), 'BalabanJ': Descriptors.BalabanJ(mol_no_h),
                'Kappa1': Descriptors.Kappa1(mol), 'Kappa2': Descriptors.Kappa2(mol), 'Kappa3': Descriptors.Kappa3(mol),
                'MaxEStateIndex': Descriptors.MaxEStateIndex(mol), 'MinEStateIndex': Descriptors.MinEStateIndex(mol),
                'MaxAbsEStateIndex': Descriptors.MaxAbsEStateIndex(mol), 'MinAbsEStateIndex': Descriptors.MinAbsEStateIndex(mol),
                'fr_NH0': Descriptors.fr_NH0(mol), 'fr_NH1': Descriptors.fr_NH1(mol), 'fr_NH2': Descriptors.fr_NH2(mol),
                'fr_N_O': Descriptors.fr_N_O(mol), 'fr_Ndealkylation1': Descriptors.fr_Ndealkylation1(mol),
                'fr_Ndealkylation2': Descriptors.fr_Ndealkylation2(mol), 'fr_C_O': Descriptors.fr_C_O(mol),
                'fr_C_O_noCOO': Descriptors.fr_C_O_noCOO(mol), 'fr_benzene': Descriptors.fr_benzene(mol),
                'fr_phenol': Descriptors.fr_phenol(mol), 'VSA_EState1': Descriptors.VSA_EState1(mol),
                'VSA_EState2': Descriptors.VSA_EState2(mol), 'VSA_EState3': Descriptors.VSA_EState3(mol),
                'SlogP_VSA1': Descriptors.SlogP_VSA1(mol), 'SlogP_VSA2': Descriptors.SlogP_VSA2(mol),
                'SlogP_VSA3': Descriptors.SlogP_VSA3(mol), 'SMR_VSA1': Descriptors.SMR_VSA1(mol),
                'SMR_VSA2': Descriptors.SMR_VSA2(mol), 'SMR_VSA3': Descriptors.SMR_VSA3(mol),
                'PEOE_VSA1': Descriptors.PEOE_VSA1(mol), 'PEOE_VSA2': Descriptors.PEOE_VSA2(mol),
                'PEOE_VSA3': Descriptors.PEOE_VSA3(mol),
            }
            for key, value in desc_dict.items():
                if pd.isna(value) or np.isinf(value):
                    desc_dict[key] = 0.0
            return {name: desc_dict.get(name, 0.0) for name in self.descriptor_names}
        except Exception:
            return None

    def generate_fingerprints(self, mol):
        if mol is None: return None
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self.fp_size)
            arr = np.zeros((self.fp_size,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except Exception:
            return None

    def extract_all_features(self, smiles_list):
        features, valid_indices, descriptors_list = [], [], []
        self.failed_molecules = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, smi in enumerate(smiles_list):
            mol = self.smiles_to_mol(smi)
            if mol is None:
                self.failed_molecules.append({'index': i, 'smiles': smi})
                continue
                
            desc = self.calculate_molecular_descriptors(mol)
            fp = self.generate_fingerprints(mol)
            
            if desc is None or fp is None:
                self.failed_molecules.append({'index': i, 'smiles': smi})
                continue

            features.append(list(desc.values()) + list(fp))
            descriptors_list.append(desc)
            valid_indices.append(i)
            
            progress_percentage = (i + 1) / len(smiles_list)
            progress_bar.progress(progress_percentage)
            status_text.text(f"🔬 Processing molecule {i+1}/{len(smiles_list)}")
            
        status_text.text("✅ Featurization complete!")
        return np.array(features), valid_indices, descriptors_list

class AdvancedDeepNeuralNetwork(nn.Module):
    """Enhanced DNN with the same architecture for model compatibility."""
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256, 128], dropout_rate=0.3):
        super(AdvancedDeepNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        self.output_layer = nn.Linear(prev_dim, 2)

    def forward(self, x):
        for layer, bn, dropout in zip(self.layers, self.batch_norms, self.dropouts):
            x = layer(x)
            x = bn(x)
            x = F.relu(x)
            x = dropout(x)
        return self.output_layer(x)

@st.cache_resource
def load_model_from_hub(model_choice, repo_id="naren-srinivas/PharmaApp"):
    """Load models from Hugging Face Hub with enhanced error handling."""
    with st.spinner(f"🚀 Loading {model_choice} model from Hugging Face..."):
        if model_choice == 'Extra Trees':
            try:
                model_path = hf_hub_download(repo_id=repo_id, filename="best_bbbp_model_Extra_Trees.joblib")
                model_data = joblib.load(model_path)
                return model_data['model'], model_data['scaler']
            except Exception as e:
                st.error(f"❌ Error loading Extra Trees model: {e}")
                return None, None
                
        elif model_choice == 'DNN (PyTorch)':
            try:
                model_path = hf_hub_download(repo_id=repo_id, filename="final_dnn_model.pth")
                scaler_path = hf_hub_download(repo_id=repo_id, filename="dnn_scaler.joblib")
                
                input_dim = 2048 + 47
                model = AdvancedDeepNeuralNetwork(input_dim)
                model.load_state_dict(torch.load(model_path))
                model.eval()
                scaler = joblib.load(scaler_path)
                return model, scaler
            except Exception as e:
                st.error(f"❌ Error loading DNN model: {e}")
                return None, None

def mol_to_image(smiles, size=(300, 300)):
    """Convert SMILES to molecular structure image."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=size)
        return img
    except:
        return None

def create_molecular_dashboard(results_df, descriptors_list):
    """Create an enhanced dashboard for molecular analysis."""
    
    st.markdown("## 📊 Molecular Analysis Dashboard")
    
    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    
    total_molecules = len(results_df)
    permeable_count = len(results_df[results_df['Prediction'] == 'Permeable'])
    non_permeable_count = total_molecules - permeable_count
    avg_probability = results_df['Probability_Permeable'].mean()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🧪 Total Molecules</h3>
            <h2>{total_molecules}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>✅ Permeable</h3>
            <h2>{permeable_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>❌ Non-Permeable</h3>
            <h2>{non_permeable_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Avg Probability</h3>
            <h2>{avg_probability:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    # Enhanced visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Donut chart for predictions
        fig_donut = go.Figure(data=[go.Pie(
            labels=['Permeable', 'Non-Permeable'],
            values=[permeable_count, non_permeable_count],
            hole=.3,
            marker=dict(colors=['#4CAF50', '#f44336'])
        )])
        fig_donut.update_layout(
            title="🎯 Prediction Distribution",
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig_donut, use_container_width=True)
    
    with col2:
        # Probability distribution
        fig_hist = px.histogram(
            results_df, 
            x='Probability_Permeable',
            nbins=20,
            title="📊 Probability Distribution",
            color_discrete_sequence=['#667eea']
        )
        fig_hist.update_layout(
            xaxis_title="Permeability Probability",
            yaxis_title="Count",
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Molecular properties analysis
    if descriptors_list:
        st.markdown("### 🔬 Molecular Properties Analysis")
        
        # Create descriptors dataframe
        desc_df = pd.DataFrame(descriptors_list)
        desc_df['Prediction'] = results_df['Prediction'].values
        
        # Key descriptors visualization
        key_descriptors = ['MW', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA', 'NumRotatableBonds']
        
        fig_props = make_subplots(
            rows=2, cols=3,
            subplot_titles=key_descriptors,
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        for i, desc in enumerate(key_descriptors):
            row = i // 3 + 1
            col = i % 3 + 1
            
            permeable_vals = desc_df[desc_df['Prediction'] == 'Permeable'][desc]
            non_permeable_vals = desc_df[desc_df['Prediction'] == 'Non-Permeable'][desc]
            
            fig_props.add_trace(
                go.Box(y=permeable_vals, name='Permeable', marker_color='#4CAF50', showlegend=(i==0)),
                row=row, col=col
            )
            fig_props.add_trace(
                go.Box(y=non_permeable_vals, name='Non-Permeable', marker_color='#f44336', showlegend=(i==0)),
                row=row, col=col
            )
        
        fig_props.update_layout(
            title="📋 Molecular Properties by Permeability",
            height=600,
            showlegend=True
        )
        st.plotly_chart(fig_props, use_container_width=True)

def create_individual_molecule_cards(results_df, smiles_col):
    """Create individual cards for each molecule with structure and details."""
    
    st.markdown("## 🧬 Individual Molecule Analysis")
    
    # Pagination
    molecules_per_page = 6
    total_pages = (len(results_df) - 1) // molecules_per_page + 1
    
    if total_pages > 1:
        page = st.selectbox("Select Page", range(1, total_pages + 1))
        start_idx = (page - 1) * molecules_per_page
        end_idx = min(start_idx + molecules_per_page, len(results_df))
        page_df = results_df.iloc[start_idx:end_idx]
    else:
        page_df = results_df
    
    # Create molecule cards
    for idx, row in page_df.iterrows():
        smiles = row[smiles_col]
        prediction = row['Prediction']
        probability = row['Probability_Permeable']
        
        # Create expandable section for each molecule
        with st.expander(f"🧪 Molecule {idx + 1}: {prediction} (Prob: {probability:.3f})"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Molecular structure
                img = mol_to_image(smiles)
                if img:
                    st.image(img, caption="Molecular Structure", use_container_width=True)
                else:
                    st.error("Could not generate molecular structure")
            
            with col2:
                # Molecule details
                card_class = "permeable-card" if prediction == "Permeable" else "non-permeable-card"
                
                st.markdown(f"""
                <div class="prediction-card {card_class}">
                    <h3>📋 Molecule Details</h3>
                    <p><strong>SMILES:</strong> {smiles}</p>
                    <p><strong>Prediction:</strong> {prediction}</p>
                    <p><strong>Probability:</strong> {probability:.4f}</p>
                    <p><strong>Confidence:</strong> {'High' if abs(probability - 0.5) > 0.3 else 'Medium' if abs(probability - 0.5) > 0.1 else 'Low'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Permeability Probability"},
                    gauge = {
                        'axis': {'range': [None, 1]},
                        'bar': {'color': "#667eea"},
                        'steps': [
                            {'range': [0, 0.5], 'color': "#ffcccb"},
                            {'range': [0.5, 1], 'color': "#90EE90"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.5
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

def main():
    """Enhanced main function with improved UI."""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Blood-Brain Barrier Permeability Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced AI-powered prediction of molecular permeability across the blood-brain barrier</p>', unsafe_allow_html=True)

    # Sidebar with enhanced styling
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    model_choice = st.sidebar.selectbox(
        "🤖 Choose Prediction Model:", 
        ['Extra Trees', 'DNN (PyTorch)'],
        help="Select the machine learning model for prediction"
    )
    
    st.sidebar.markdown("### 📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file with SMILES", 
        type=["csv"],
        help="Upload a CSV file containing SMILES strings"
    )
    
    # Information panel
    with st.sidebar.expander("ℹ️ About This Tool"):
        st.markdown("""
        This tool predicts whether molecules can cross the blood-brain barrier using:
        
        - **Molecular Descriptors**: Chemical properties
        - **Morgan Fingerprints**: Structural features  
        - **Machine Learning**: Trained models
        
        **Input Requirements:**
        - CSV file with SMILES column
        - NO prediction/label columns needed
        
        The tool will predict P (Permeable) or NP (Non-Permeable) for you!
        """)

    if uploaded_file is not None:
        # Enhanced file reading with better error handling
        df = None
        encodings = ['utf-8', 'latin1', 'cp1252']
        
        for encoding in encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding)
                st.sidebar.success(f"✅ File loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
                return

        if df is None:
            st.error("❌ Could not read the CSV file. Please check the file format.")
            return
            
        # Find SMILES column
        smiles_col = None
        for col in df.columns:
            if 'smiles' in col.lower():
                smiles_col = col
                break
        
        if smiles_col is None:
            st.error("❌ No SMILES column found. Please ensure your CSV has a column containing 'smiles' in its name.")
            return
        
        st.sidebar.success(f"🎯 Found SMILES column: '{smiles_col}'")
        
        # Data preview
        st.markdown("## 📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Enhanced prediction button
        if st.button(f"🚀 Run Prediction with {model_choice}", type="primary"):
            
            # Load model
            model, scaler = load_model_from_hub(model_choice)
            if model is None:
                return

            # Feature extraction
            smiles_list = df[smiles_col].dropna().tolist()
            
            with st.spinner("🔬 Extracting molecular features..."):
                feature_gen = MolecularFeatureGenerator()
                X, valid_indices, descriptors_list = feature_gen.extract_all_features(smiles_list)
                
            if X.shape[0] == 0:
                st.warning("⚠️ No valid molecules could be processed.")
                return
            
            # Make predictions
            with st.spinner("🤖 Making predictions..."):
                scaler = StandardScaler().fit(X)
                X_scaled = scaler.transform(X)
                
                if model_choice == 'Extra Trees':
                    preds = model.predict(X_scaled)
                    probs = model.predict_proba(X_scaled)[:, 1]
                else:  # DNN
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X_scaled)
                        outputs = model(X_tensor)
                        probs = torch.softmax(outputs, dim=1)[:, 1].numpy()
                        preds = (probs > 0.5).astype(int)

            # Prepare results
            results_df = df.iloc[valid_indices].copy()
            results_df['Prediction'] = ['Permeable' if p == 1 else 'Non-Permeable' for p in preds]
            results_df['Probability_Permeable'] = probs
            results_df['Confidence'] = ['High' if abs(p - 0.5) > 0.3 else 'Medium' if abs(p - 0.5) > 0.1 else 'Low' for p in probs]
            
            # Display results
            st.success("✅ Predictions completed successfully!")
            
            # Enhanced dashboard
            create_molecular_dashboard(results_df, descriptors_list)
            
            # Results table
            st.markdown("## 📊 Results Table")
            st.dataframe(
                results_df[['Prediction', 'Probability_Permeable', 'Confidence', smiles_col]],
                use_container_width=True
            )
            
            # Download button
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download Results as CSV",
                data=csv,
                file_name=f"bbb_predictions_{model_choice.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                type="primary"
            )
            
            # Individual molecule analysis
            create_individual_molecule_cards(results_df, smiles_col)
            
            # Failed molecules
            if feature_gen.failed_molecules:
                with st.expander("⚠️ Failed Molecules"):
                    st.warning(f"Could not process {len(feature_gen.failed_molecules)} molecules:")
                    failed_df = pd.DataFrame(feature_gen.failed_molecules)
                    st.dataframe(failed_df, use_container_width=True)

    else:
        # Welcome screen
        st.markdown("""
        ### 🎯 Getting Started
        
        1. **Upload your CSV file** using the sidebar
        2. **Ensure it contains a SMILES column** (column name should contain 'smiles')
        3. **Do NOT include any target/label columns** (the tool will predict P/NP for you)
        4. **Select your prediction model** (Extra Trees or DNN)
        5. **Click 'Run Prediction'** to analyze your molecules
        
        ### 📊 What you'll get:
        - **Permeability predictions** for each molecule
        - **Probability scores** indicating confidence
        - **Interactive visualizations** of results
        - **Molecular structure images** for each compound
        - **Detailed analysis** of molecular properties
        """)
        
        # Example data format
        st.markdown("### 📋 Example Input Data Format")
        st.info("⚠️ **Important**: Your CSV should NOT contain any prediction/label columns (like 'permeable', 'P/NP', etc.). The tool will predict these values for you!")
        
        example_df = pd.DataFrame({
            'compound_id': ['COMP001', 'COMP002', 'COMP003'],
            'smiles': ['CCO', 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'],
            'compound_name': ['Ethanol', 'Ibuprofen', 'Caffeine']
        })
        st.dataframe(example_df, use_container_width=True)
        
        st.markdown("""
        **Required:**
        - `smiles` column (or any column with 'smiles' in the name)
        
        **Optional:**
        - Identifier columns (compound_id, name, etc.)
        - Any other descriptive columns
        
        **❌ Do NOT include:**
        - Target labels (permeable/non-permeable, P/NP, etc.)
        - Any prediction columns
        """)

if __name__ == '__main__':
    main()