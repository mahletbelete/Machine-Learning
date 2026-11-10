
"""
Practical Engineering Case Study:
Pump Condition Classification

Algorithms:
1. Decision Tree
2. Random Forest
3. Support Vector Machine (SVM)
4. Gaussian Naive Bayes
5. AdaBoost
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ============================================================
# PART 1. CREATE DATASET (30 PUMP RECORDS)
# ============================================================

data = [
    # ID, Vibration, RPM, Temperature, Pressure, Operating_Hours, Condition

    # NORMAL (10 records)
    [1,  0.8, 1490, 45, 5.8,  800, "Normal"],
    [2,  1.1, 1485, 49, 5.6, 1200, "Normal"],
    [3,  1.3, 1500, 52, 5.9, 1500, "Normal"],
    [4,  1.5, 1475, 55, 5.5, 1800, "Normal"],
    [5,  1.0, 1495, 48, 6.0, 1000, "Normal"],
    [6,  1.7, 1465, 58, 5.3, 2200, "Normal"],
    [7,  1.2, 1480, 51, 5.7, 1600, "Normal"],
    [8,  1.8, 1455, 62, 5.1, 2600, "Normal"],
    [9,  1.4, 1470, 56, 5.4, 2000, "Normal"],
    [10, 0.9, 1505, 46, 6.1,  900, "Normal"],

    # WARNING (10 records)
    [11, 2.5, 1450, 68, 4.9, 3200, "Warning"],
    [12, 2.8, 1435, 71, 4.7, 3800, "Warning"],
    [13, 3.0, 1420, 73, 4.5, 4200, "Warning"],
    [14, 2.3, 1460, 66, 5.0, 3000, "Warning"],
    [15, 3.4, 1405, 76, 4.3, 4800, "Warning"],
    [16, 2.7, 1440, 70, 4.8, 3500, "Warning"],
    [17, 3.6, 1395, 79, 4.1, 5200, "Warning"],
    [18, 2.9, 1415, 74, 4.6, 4400, "Warning"],
    [19, 3.2, 1400, 77, 4.2, 5000, "Warning"],
    [20, 2.4, 1455, 67, 4.9, 3300, "Warning"],

    # FAULT (10 records)
    [21, 5.2, 1350, 85, 3.5, 5800, "Fault"],
    [22, 6.1, 1320, 91, 3.1, 6500, "Fault"],
    [23, 7.0, 1285, 96, 2.8, 7400, "Fault"],
    [24, 5.8, 1335, 88, 3.3, 6200, "Fault"],
    [25, 7.5, 1260, 102, 2.5, 8500, "Fault"],
    [26, 6.5, 1300, 94, 2.9, 7100, "Fault"],
    [27, 4.9, 1365, 83, 3.7, 5500, "Fault"],
    [28, 7.8, 1240, 105, 2.3, 9000, "Fault"],
    [29, 5.5, 1340, 87, 3.4, 6000, "Fault"],
    [30, 6.8, 1290, 98, 2.7, 7800, "Fault"]
]

columns = [
    "Pump_ID",
    "Vibration",
    "RPM",
    "Temperature",
    "Pressure",
    "Operating_Hours",
    "Condition"
]

df = pd.DataFrame(data, columns=columns)

print("\nPUMP CONDITION DATASET")
print(df.to_string(index=False))

print("\nCLASS DISTRIBUTION")
print(df["Condition"].value_counts())


# ============================================================
# PART 2. DATA VISUALIZATION
# ============================================================

sns.set_theme(style="whitegrid")

# Vibration vs Temperature
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Temperature",
    y="Vibration",
    hue="Condition",
    style="Condition",
    s=100
)

plt.title("Pump Temperature vs Vibration")
plt.xlabel("Temperature (°C)")
plt.ylabel("Vibration (mm/s)")
plt.tight_layout()
plt.show()


# Correlation matrix
plt.figure(figsize=(9, 6))

sns.heatmap(
    df.drop(columns=["Pump_ID", "Condition"]).corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Pump Sensor Correlation Matrix")
plt.tight_layout()
plt.show()


# ============================================================
# PART 3. SEPARATE FEATURES AND TARGET
# ============================================================

feature_cols = [
    "Vibration",
    "RPM",
    "Temperature",
    "Pressure",
    "Operating_Hours"
]

X = df[feature_cols]
y = df["Condition"]

class_labels = ["Normal", "Warning", "Fault"]


# ============================================================
# PART 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTRAINING RECORDS:", len(X_train))
print("TESTING RECORDS:", len(X_test))


# ============================================================
# PART 5. DEFINE FIVE CLASSIFICATION MODELS
# ============================================================

# 1. Support Vector Machine
svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale"))
])


# 2. Gaussian Naive Bayes
naive_bayes_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GaussianNB())
])


# 3. Decision Tree
decision_tree_model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)


# 4. Random Forest
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=42
)


# 5. AdaBoost
adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    ),
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)


models = {
    "SVM": svm_model,
    "Naive Bayes": naive_bayes_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "AdaBoost": adaboost_model
}


# ============================================================
# PART 6. TRAIN AND EVALUATE MODELS
# ============================================================

results = []
predictions = {}

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(model_name.upper())
    print("=" * 60)

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    predictions[model_name] = y_pred

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    # Fault-specific evaluation
    fault_precision = precision_score(
        y_test, y_pred,
        labels=["Fault"],
        average=None,
        zero_division=0
    )[0]

    fault_recall = recall_score(
        y_test, y_pred,
        labels=["Fault"],
        average=None,
        zero_division=0
    )[0]

    results.append({
        "Algorithm": model_name,
        "Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1,
        "Fault Precision": fault_precision,
        "Fault Recall": fault_recall
    })

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro Precision: {precision:.3f}")
    print(f"Macro Recall: {recall:.3f}")
    print(f"Macro F1-score: {f1:.3f}")

    print(f"Fault Precision: {fault_precision:.3f}")
    print(f"Fault Recall: {fault_recall:.3f}")

    print("\nCLASSIFICATION REPORT")

    print(classification_report(
        y_test,
        y_pred,
        labels=class_labels,
        zero_division=0
    ))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=class_labels,
        display_labels=class_labels,
        cmap="Blues",
        colorbar=False,
        ax=ax
    )

    ax.set_title(f"{model_name} Confusion Matrix")

    plt.tight_layout()
    plt.show()


# ============================================================
# PART 7. COMPARE FIVE ALGORITHMS
# ============================================================

results_df = pd.DataFrame(results).set_index("Algorithm")

print("\nMODEL PERFORMANCE COMPARISON")

print(results_df.round(3))


# Comparison graph
results_df.plot(
    kind="bar",
    figsize=(12, 6),
    ylim=(0, 1.05)
)

plt.title("Comparison of Five Classification Algorithms")
plt.ylabel("Evaluation Score")
plt.xlabel("Machine Learning Algorithm")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# ============================================================
# PART 8. FEATURE IMPORTANCE
# ============================================================

tree_models = {
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}

for name, model in tree_models.items():

    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    print(f"\n{name} FEATURE IMPORTANCE")

    print(importance.round(3))

    plt.figure(figsize=(8, 5))

    importance.sort_values().plot(kind="barh")

    plt.title(f"{name} Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Pump Sensor Features")

    plt.tight_layout()
    plt.show()


# Visualize Decision Tree
plt.figure(figsize=(18, 9))

plot_tree(
    decision_tree_model,
    feature_names=feature_cols,
    class_names=class_labels,
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Pump Condition Decision Tree")

plt.tight_layout()
plt.show()


# ===========================================================
# PART 9. TEST A NEW PUMP OBSERVATION
# ===========================================================

new_pump = pd.DataFrame([{
    "Vibration": 6.2,
    "RPM": 1310,
    "Temperature": 93,
    "Pressure": 2.9,
    "Operating_Hours": 6800
}])

print("\nNEW PUMP OBSERVATION")
print(new_pump.to_string(index=False))

print("\nPREDICTION RESULTS")

for name, model in models.items():

    predicted_class = model.predict(new_pump)[0]

    print(f"{name:15s} prediction: {predicted_class}")

# ===========================================================
# PART 10. FINAL RESULT
#============================================================

best_model = results_df["Macro F1"].idxmax()

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print("Best model based on Macro F1-score:", best_model)

print("\nEND OF PUMP CONDITION CLASSIFICATION")

