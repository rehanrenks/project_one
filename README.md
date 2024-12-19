# project_one
# Hi there, I'm Rehan Renks 👋

Welcome to my GitHub profile! I'm a passionate developer with a keen interest in biomedical data analysis and bioinformatics. One of my key projects involves analyzing biomarkers related to Blood-Brain Barrier (BBB) dysfunction and Alzheimer's disease. Let's connect and explore the fascinating world of bioinformatics together!

## 🧠 Alzheimer's Disease Biomarker Analysis Project

In this project, I focus on analyzing various biomarkers that are crucial in understanding BBB dysfunction and early detection of Alzheimer's disease. The biomarkers analyzed include:

- **Claudin-5**: A protein found in the tight junctions between the endothelial cells of the BBB. Decreased levels in the blood can indicate early BBB breakdown.
- **MMP-9 (Matrix Metalloproteinase-9)**: An enzyme that breaks down extracellular matrix components, increasing BBB permeability. Elevated levels are associated with early Alzheimer's progression.
- **S100B**: A protein produced by astrocytes. When the BBB breaks down, S100B enters the bloodstream, indicating neuroinflammation and Alzheimer's pathology.
- **miR-124**: A microRNA involved in regulating brain inflammation. Increased levels in the blood correlate with neurodegeneration and BBB breakdown.
- **miR-155**: Another microRNA linked to inflammation and BBB dysfunction. Higher levels suggest early brain damage, making it a valuable biomarker for Alzheimer's prediction.

## 🛠️ Technologies & Tools

- **Languages**: Python
- **Libraries**: Pandas, NumPy, SciPy, Matplotlib, Seaborn
- **Tools**: Jupyter Notebook, Git, GitHub

## 🔬 Project Goals

1. **Data Collection**: Gather and preprocess data on the mentioned biomarkers from various sources.
2. **Statistical Analysis**: Perform statistical tests to identify significant changes in biomarker levels.
3. **Visualization**: Create informative visualizations to represent the data and findings.
4. **Predictive Modeling**: Develop machine learning models to predict BBB dysfunction and early Alzheimer's based on biomarker levels.

## 📈 Example Analysis

Here's a snippet of Python code used in the project to identify early signs of BBB breakdown:

```python
import pandas as pd

# Example data for biomarkers
data = {
    'PatientID': [1, 2, 3, 4, 5],
    'Claudin-5': [0.8, 0.6, 0.5, 0.3, 0.2],
    'MMP-9': [50, 70, 85, 90, 95],
    'S100B': [0.05, 0.08, 0.1, 0.15, 0.2],
    'miR-124': [100, 150, 200, 250, 300],
    'miR-155': [50, 60, 70, 80, 90]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Function to identify early signs of BBB breakdown and Alzheimer's
def identify_bbb_dysfunction(df):
    results = []
    for index, row in df.iterrows():
        patient_id = row['PatientID']
        claudin_5 = row['Claudin-5']
        mmp_9 = row['MMP-9']
        s100b = row['S100B']
        mir_124 = row['miR-124']
        mir_155 = row['miR-155']
        
        bbb_dysfunction = claudin_5 < 0.5 or mmp_9 > 80 or s100b > 0.1 or mir_124 > 200 or mir_155 > 70
        result = {
            'PatientID': patient_id,
            'BBB_Dysfunction': bbb_dysfunction
        }
        results.append(result)
    
    return pd.DataFrame(results)

# Identify BBB dysfunction in the dataset
bbb_results = identify_bbb_dysfunction(df)

# Print results
print(bbb_results)