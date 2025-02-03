# Bond Portfolio Analysis Application
# Application d'Analyse de Portefeuille Obligataire

[English](#english) | [France](#france)

# English

## Features

### 1. Portfolio Global Analysis
- Investment Grade, High Yield, and Cash allocation
- Sector weight visualization
- Rating distribution
- Payment rank analysis

### 2. Performance Metrics
- Yield to Maturity
- Modified Duration
- Duration Contribution
- Yield Contribution
- RPAP (Duration/Yield Contribution Ratio)

### 3. Visualizations
- Sector allocation charts
- Rating distribution
- Rating box plots
- Future cash flow analysis
- Cross matrices (Sector x Duration, Country x Duration)

### 4. RPAP Analysis
The application includes detailed RPAP analysis (Duration/Yield Contribution Ratio):
- RPAP distribution
- Analysis by rating
- Descriptive statistics
- Interpretation:
  * RPAP = 1: equal contribution to duration and yield
  * RPAP < 1: better efficiency (more yield contribution than duration)
  * RPAP > 1: lower efficiency (more duration contribution than yield)

## Project Structure
- `app_mandats.py`: Main Streamlit application
- `mandats_bonds_new.py`: Bond data generation script

## Prerequisites
```
pandas
numpy
plotly
streamlit
```

## Installation
1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Launch the application:
```bash
streamlit run app_mandats.py
```

2. The interface allows you to:
   - Select a specific mandate
   - View different metrics
   - Explore detailed analyses via expanders
   - Sort and filter data in the main table

## Adding a New Column in app_mandats.py

To add a new column (e.g., "Dirty Price"):

1. In `app_mandats.py`:
   ```python
   # After loading the data
   bonds_df['dirty_price'] = ...  # Add your calculation here
   
   # In the dataframe configuration
   st.dataframe(
       bonds_df,
       column_config={
           # ... other columns ...
           "dirty_price": "Dirty Price",  # Add this line
       }
   )
   ```

2. To include the column in visualizations:
   ```python
   # In the charts section
   fig = px.scatter(
       bonds_df,
       x='dirty_price',  # Use the new column
       y='YAS_MOD_DUR',
       title='Duration vs Dirty Price'
   )
   ```

3. For weighted averages:
   ```python
   # Calculate weighted average
   avg_dirty_price = np.average(
       bonds_df['dirty_price'],
       weights=bonds_df['Poids en %']
   )
   ```

Note: Make sure to handle any necessary data preprocessing or missing values for the new column.

---

# France

## Fonctionnalités

### 1. Analyse Globale du Portefeuille
- Répartition entre Investment Grade, High Yield et Liquidités
- Visualisation des poids par secteur
- Distribution des ratings
- Analyse des rangs de paiement

### 2. Métriques de Performance
- Yield to Maturity
- Duration Modifiée
- Contribution à la Duration
- Contribution au Yield
- RPAP (Rapport entre Contributions Duration/Yield)

### 3. Visualisations
- Graphiques de répartition sectorielle
- Distribution des ratings
- Box plots par rating
- Analyse des cash flows futurs
- Matrices de croisement (Secteur x Duration, Pays x Duration)

### 4. Analyse RPAP
L'application inclut une analyse détaillée du RPAP (Rapport entre Contributions Duration/Yield) :
- Distribution des RPAP
- Analyse par rating
- Statistiques descriptives
- Interprétation :
  * RPAP = 1 : contribution équivalente à la duration et au yield
  * RPAP < 1 : meilleure efficience (plus de contribution au yield qu'à la duration)
  * RPAP > 1 : moindre efficience (plus de contribution à la duration qu'au yield)

## Structure du Projet
- `app_mandats.py` : Application principale Streamlit
- `mandats_bonds_new.py` : Script de génération des données obligataires

## Prérequis
```
pandas
numpy
plotly
streamlit
```

## Installation
1. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation
1. Lancer l'application :
```bash
streamlit run app_mandats.py
```

2. L'interface permet de :
   - Sélectionner un mandat spécifique
   - Visualiser les différentes métriques
   - Explorer les analyses détaillées via les expanders
   - Trier et filtrer les données dans le tableau principal

## Ajouter une Nouvelle Colonne dans app_mandats.py

Pour ajouter une nouvelle colonne (par exemple "Dirty Price") :

1. Dans `app_mandats.py` :
   ```python
   # Après le chargement des données
   bonds_df['dirty_price'] = ...  # Ajouter votre calcul ici
   
   # Dans la configuration du dataframe
   st.dataframe(
       bonds_df,
       column_config={
           # ... autres colonnes ...
           "dirty_price": "Dirty Price",  # Ajouter cette ligne
       }
   )
   ```

2. Pour inclure la colonne dans les visualisations :
   ```python
   # Dans la section des graphiques
   fig = px.scatter(
       bonds_df,
       x='dirty_price',  # Utiliser la nouvelle colonne
       y='YAS_MOD_DUR',
       title='Duration vs Dirty Price'
   )
   ```

3. Pour les moyennes pondérées :
   ```python
   # Calculer la moyenne pondérée
   avg_dirty_price = np.average(
       bonds_df['dirty_price'],
       weights=bonds_df['Poids en %']
   )
   ```

Note : Assurez-vous de gérer tout prétraitement de données ou valeurs manquantes nécessaires pour la nouvelle colonne.
