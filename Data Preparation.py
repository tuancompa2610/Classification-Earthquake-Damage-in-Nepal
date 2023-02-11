# Import Library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from category_encoders import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils.validation import check_is_fitted
from category_encoders import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load data. Since our csv file has Unamed columns we should add parameter "index_col" in read_csv function
df = pd.read_csv("/content/nepal_earthquake.csv", index_col = 0)

df["damage_grade"].value_counts().sort_values()
df["damage_grade"] = df["damage_grade"].str[-1].astype(int)
df["severe_damage"] = (df["damage_grade"] > 3).astype(int)

# Set "building_id" as index and drop those unnecessary columns 
df.drop(columns = ["b_id", "damage_grade"], inplace = True)
df = df.set_index("building_id")
df.head()



