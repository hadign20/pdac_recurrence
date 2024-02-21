import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
from skfeature.function.information_theoretical_based import MRMR
from sklearn.decomposition import PCA

#===========================================================
# parameters
excel_file_name = "../TextureFeatures_Hadi.xlsx"
sheetName = "LiverFeatures"
output_var_name = 'EarlyRecurrence'
extra_columns = ['CaseNo']
correlation_thresh = 0.8
num_of_features = 5
feature_selection_method = "MRMR"  # MRMR, PCA


#===========================================================
# preprocessing
df = pd.read_excel(excel_file_name, sheet_name=sheetName)
df.drop(columns=extra_columns, inplace=True)

#===========================================================
# train test split
x = df.drop(columns=[output_var_name])
y = df[output_var_name]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#===========================================================
# remove correlated features
corr_matrix = x_train.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
highly_correlated_cols = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_thresh)]

x_train.drop(columns=highly_correlated_cols, inplace=True)
x_test.drop(columns=highly_correlated_cols, inplace=True)


#===========================================================
# feature selection
if feature_selection_method == "MRMR":
    # Selects features using MRMR (Minimum Redundancy Maximum Relevance).
    selected_features = MRMR.mrmr(x_train.values, y_train.values, mode="index", n_selected_features=num_of_features)
    x_train = x_train.iloc[:, selected_features]
    x_test = x_test.iloc[:, selected_features]

elif feature_selection_method == "PCA":
    # Selects features using PCA (Principal Component Analysis).
    pca = PCA(n_components=num_of_features)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    # Transforming back to original feature space
    selected_feature_indices = np.abs(pca.components_).argsort()[-num_of_features:][::-1]
    selected_feature_names = x.columns[selected_feature_indices]

    x_train = pd.DataFrame(x_train, columns=selected_feature_names)
    x_test = pd.DataFrame(x_test, columns=selected_feature_names)

else:
    print("wrong feature selection method...")


#===========================================================
# model building
# Train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(x_train, y_train)

# Predict on test set
y_pred_rf = rf_classifier.predict(x_test)
y_proba_rf = rf_classifier.predict_proba(x_test)[:, 1]

# Compute AUC
auc_rf = roc_auc_score(y_test, y_proba_rf)

# Compute ROC curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

# Compute confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)

# Compute sensitivity and specificity
tn, fp, fn, tp = cm_rf.ravel()
sensitivity_rf = tp / (tp + fn)
specificity_rf = tn / (tn + fp)
ppv_rf = tp / (tp + fp)
npv_rf = tn / (tn + fn)


# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc_rf)
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
plt.savefig('rf_roc_curve.png')



# Get feature importances
feature_importances_rf = rf_classifier.feature_importances_

plt.figure(figsize=(10, 6))
plt.barh(x_train.columns, feature_importances_rf)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance - Random Forest')
plt.show()
plt.savefig('rf_feature_importance.png')

















from sklearn.linear_model import LogisticRegression

# Train Logistic Regression classifier
logreg_classifier = LogisticRegression(max_iter=1000, random_state=42)
logreg_classifier.fit(x_train, y_train)

# Predict on test set
y_pred_logreg = logreg_classifier.predict(x_test)
y_proba_logreg = logreg_classifier.predict_proba(x_test)[:, 1]

# Compute AUC
auc_logreg = roc_auc_score(y_test, y_proba_logreg)

# Compute ROC curve
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_proba_logreg)

# Compute confusion matrix
cm_logreg = confusion_matrix(y_test, y_pred_logreg)

# Compute sensitivity and specificity
tn, fp, fn, tp = cm_logreg.ravel()
sensitivity_logreg = tp / (tp + fn)
specificity_logreg = tn / (tn + fp)
ppv_logreg = tp / (tp + fp)
npv_logreg = tn / (tn + fn)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_logreg, tpr_logreg, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc_logreg)
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Logistic Regression')
plt.legend(loc="lower right")

# Save ROC curve plot
plt.savefig('roc_curve_logreg.png')














from sklearn.svm import SVC

# Train SVM classifier
svm_classifier = SVC(probability=True, random_state=42)
svm_classifier.fit(x_train, y_train)

# Predict on test set
y_pred_svm = svm_classifier.predict(x_test)
y_proba_svm = svm_classifier.predict_proba(x_test)[:, 1]

# Compute AUC
auc_svm = roc_auc_score(y_test, y_proba_svm)

# Compute ROC curve
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)

# Compute confusion matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)

# Compute sensitivity and specificity
tn, fp, fn, tp = cm_svm.ravel()
sensitivity_svm = tp / (tp + fn)
specificity_svm = tn / (tn + fp)
ppv_svm = tp / (tp + fp)
npv_svm = tn / (tn + fn)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_svm, tpr_svm, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc_svm)
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - SVM')
plt.legend(loc="lower right")

# Save ROC curve plot
plt.savefig('roc_curve_svm.png')
