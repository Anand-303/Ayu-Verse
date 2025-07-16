import pandas as pd
import json
from collections import defaultdict
import re

# Load and process the dataset
class HerbalDataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = self._load_dataset()
        self.medical_conditions = self._extract_conditions()
        self.plant_index = self._create_plant_index()
        self.condition_index = self._create_condition_index()

    def _load_dataset(self):
        """Load the dataset and clean it"""
        try:
            df = pd.read_csv(self.dataset_path)
            # Convert all text columns to lowercase
            text_columns = ['Plant Name', 'Medical Condition Treated', 'Part Used', 'Usage Method']
            for col in text_columns:
                df[col] = df[col].str.lower()
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def _extract_conditions(self):
        """Extract and clean medical conditions"""
        conditions = set()
        for condition in self.data['Medical Condition Treated']:
            # Split conditions by commas and add to set
            for cond in str(condition).split(', '):
                # Clean condition text
                cond = re.sub(r'[^a-z\s]', '', cond.lower())
                conditions.add(cond.strip())
        return list(conditions)

    def _create_plant_index(self):
        """Create an index of plants by their properties"""
        plant_index = defaultdict(dict)
        for _, row in self.data.iterrows():
            plant_name = row['Plant Name'].lower()
            condition = row['Medical Condition Treated'].lower()
            
            if plant_name not in plant_index:
                plant_index[plant_name] = {
                    'scientific_name': row['Scientific Name'],
                    'parts_used': [],
                    'conditions': [],
                    'usage_methods': [],
                    'dosages': [],
                    'side_effects': [],
                    'nutritional_benefits': [],
                    'drug_interactions': []
                }
            
            # Add condition if not already present
            if condition not in plant_index[plant_name]['conditions']:
                plant_index[plant_name]['conditions'].append(condition)
                
            # Add other properties
            plant_index[plant_name]['parts_used'].append(row['Part Used'])
            plant_index[plant_name]['usage_methods'].append(row['Usage Method'])
            plant_index[plant_name]['dosages'].append(row['Dosage'])
            plant_index[plant_name]['side_effects'].append(row['Side Effects'])
            plant_index[plant_name]['nutritional_benefits'].append(row['Nutritional Benefits'])
            plant_index[plant_name]['drug_interactions'].append(row['Drug Interactions'])
        
        return plant_index

    def _create_condition_index(self):
        """Create an index of conditions to plants"""
        condition_index = defaultdict(list)
        for _, row in self.data.iterrows():
            condition = row['Medical Condition Treated'].lower()
            plant_name = row['Plant Name'].lower()
            condition_index[condition].append(plant_name)
        return condition_index

    def get_plant_info(self, plant_name):
        """Get detailed information about a specific plant"""
        plant_name = plant_name.lower()
        if plant_name in self.plant_index:
            return self.plant_index[plant_name]
        return None

    def get_plants_for_condition(self, condition):
        """Get all plants that treat a specific condition"""
        condition = condition.lower()
        if condition in self.condition_index:
            return self.condition_index[condition]
        return []

    def search_conditions(self, query):
        """Search for medical conditions similar to the query"""
        query = query.lower()
        matches = []
        for condition in self.medical_conditions:
            if query in condition:
                matches.append(condition)
        return matches

# Initialize the dataset processor
dataset = HerbalDataset('herbal_remedies_dataset.csv')
