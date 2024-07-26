import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from skfeature.function.information_theoretical_based import MRMR
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score, f1_score
from joblib import dump, load
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats import ranksums
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN

#===========================================================
# parameters
#===========================================================
#excel_file_name = "../TextureFeatures_Hadi.xlsx"
#sheetName = "PancreasFeatures"

filename = "JayaTexturePancreasMedian_4_5"
excel_file_name = "../jaya_excel_315/" + filename + ".xlsx"
sheetName = "PancreasFeatures"

result_folder = "./results"
output_var_name = 'EarlyRecurrence'
extra_columns = ['CaseNo']
correlation_thresh = 0.8
num_of_features = 20
feature_selection_method = "MRMR"  # MRMR, PCA


res_folder = os.path.join(result_folder, filename)
os.makedirs(res_folder, exist_ok=True)

#===========================================================
# functions
#===========================================================
def train_evaluate_model(classifier, x_train, y_train, x_test, y_test):
    classifier.fit(x_train, y_train)
    y_proba = classifier.predict_proba(x_test)[:,1]
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return fpr, tpr, auc, y_proba



def plot_roc_curve(fprs, tprs, aucs, labels):
    plt.figure(figsize=(8,6))
    for fpr, tpr, auc, label in zip(fprs, tprs, aucs, labels):
        plt.plot(fpr, tpr, label='%s (AUC = %0.2f)' % (label, auc))
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(res_folder + '/' + 'roc_curve_all_models.png')


def compute_auc_ci(y_true, y_proba, n_bootstraps=1000, ci=95):
    auc_scores = []
    n_samples = len(y_true)
    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_bootstrap = y_true.iloc[indices]
        y_proba_bootstrap = y_proba[indices]
        auc = roc_auc_score(y_true_bootstrap, y_proba_bootstrap)
        auc_scores.append(auc)
    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile
    auc_lower = np.percentile(auc_scores, lower_percentile)
    auc_upper = np.percentile(auc_scores, upper_percentile)
    auc_mean = np.mean(auc_scores)
    return auc_mean, auc_lower, auc_upper



def train_evaluate_model_with_ci(classifier, x_train, y_train, x_test, y_test):
    classifier.fit(x_train, y_train)
    y_proba = classifier.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    lower_bound, upper_bound = compute_auc_ci(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return fpr, tpr, auc, lower_bound, upper_bound

def plot_roc_curve_with_ci(fprs, tprs, aucs, lower_bounds, upper_bounds, labels):
    plt.figure(figsize=(8, 6))
    for fpr, tpr, auc, lb, ub, label in zip(fprs, tprs, aucs, lower_bounds, upper_bounds, labels):
        plt.plot(fpr, tpr, label='%s (AUC = %0.2f [CI %0.2f-%0.2f])' % (label, auc, lb, ub))
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(res_folder + '/' + 'roc_curve_all_models_with_ci.png')



def plot_feature_importance(classifier, feature_names, classifier_name):
    if hasattr(classifier, 'feature_importances_'):
        feature_importances = classifier.feature_importances_
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, feature_importances)
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title(f'Feature Importance - {classifier_name}')
        plt.savefig(res_folder + '/' + f'feature_importance_{classifier_name}.png')

    elif hasattr(classifier, 'coef_'):
        feature_importances = classifier.coef_[0]
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, feature_importances)
        plt.xlabel('Coefficient Value')
        plt.ylabel('Feature')
        plt.title(f'Feature Importance - {classifier_name}')
        plt.savefig(res_folder + '/' + f'feature_importance_{classifier_name}.png')
        plt.show()

    else:
        print(f"Feature importances not available for {classifier_name}.")


def save_metrics_table(metrics_dict, filename):
    df = pd.DataFrame(metrics_dict)
    df.to_csv(filename, index=False)



# Function to compute metrics
def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) else 0
    ppv = precision_score(y_true, y_pred)
    npv = tn / (tn + fn) if (tn + fn) else 0
    return cm, sensitivity, specificity, ppv, npv

# Function to cross-validate and compute metrics
def cross_validate_metrics(classifier, x_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_train = cross_val_predict(classifier, x_train, y_train, cv=cv)
    cm_train, sensitivity_train, specificity_train, ppv_train, npv_train = compute_metrics(y_train, y_pred_train)
    return cm_train, sensitivity_train, specificity_train, ppv_train, npv_train


def plot_confusion_matrix(cm, classifier_name):
    class_names = [0, 1]  # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    ax.xaxis.set_label_position("top")

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {classifier_name}')
    plt.savefig(res_folder + '/' + f'confusion_matrix_{classifier_name}.png')
    plt.show()





#===========================================================
# preprocessing
#===========================================================
#df = pd.read_excel(excel_file_name, sheet_name=sheetName)
df = pd.read_excel(excel_file_name)
df.drop(columns=extra_columns, inplace=True)



#===========================================================
# train test split
#===========================================================
x = df.drop(columns=[output_var_name])
y = df[output_var_name]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Resample the data
sampling_strategy={0: 182, 1: 120}
oversampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
smote_enn = SMOTEENN(random_state=42)


#x_train_oversamples, y_train_oversamples = oversampler.fit_resample(x_train, y_train)
#x_train_resampled, y_train_resampled = smote_enn.fit_resample(x_train, y_train)

#x_train, y_train = smote_enn.fit_resample(x_train, y_train)
x_train, y_train = oversampler.fit_resample(x_train, y_train)

#===========================================================
# remove correlated features
#===========================================================
corr_matrix = x_train.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
highly_correlated_cols = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_thresh)]

x_train.drop(columns=highly_correlated_cols, inplace=True)
x_test.drop(columns=highly_correlated_cols, inplace=True)

# plot correlation
f, ax = plt.subplots(figsize=(10, 8))
corr_heatmap = sns.heatmap(corr_matrix, mask=np.zeros_like(corr_matrix, dtype=np.bool_),
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
fig = corr_heatmap.get_figure()
fig.savefig(res_folder + '/' + "corr_heatmap.png")


# #===========================================================
# # Univariate Analysis
# #===========================================================
# p_values = []
# input_variable_names = x.columns
#
# for feature in input_variable_names:
#     p_value = ranksums(x[y == 0][feature], x[y == 1][feature])[1]
#     p_values.append(p_value)
#
# summary_stats = pd.DataFrame({"input_variable": input_variable_names, "p_value": p_values})
# summary_stats.sort_values(by="p_value", inplace=True)
#
# p_value_threshold1 = 0.05
# p_value_threshold2 = 0.45
#
# #selected_variables = summary_stats.loc[(summary_stats["p_value"] <= p_value_threshold1) & (summary_stats["p_value"] >= p_value_threshold2), "input_variable"]
# selected_variables = summary_stats.loc[(summary_stats["p_value"] <= p_value_threshold1) , "input_variable"]
#
# x_train = x_train[selected_variables]
# x_test = x_test[selected_variables]



#===========================================================
# feature selection
#===========================================================
if feature_selection_method == "MRMR":
    # Selects features using MRMR (Minimum Redundancy Maximum Relevance).
    selected_features = MRMR.mrmr(x_train.values, y_train.values, mode="index", n_selected_features=num_of_features)
    #selected_features = MRMR.mrmr(x_train.values, y_train.values, mode="index")
    x_train = x_train.iloc[:, selected_features]
    x_test = x_test.iloc[:, selected_features]

elif feature_selection_method == "PCA":
    # Selects features using PCA (Principal Component Analysis).
    pca = PCA(n_components=num_of_features)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    # Transforming back to original feature space
    #selected_feature_indices = np.abs(pca.components_).argsort(axis=1)[:, -num_of_features:][:, ::-1]
    #selected_feature_names = x.columns[selected_feature_indices.flatten()]

    #x_train = pd.DataFrame(x_train, columns=selected_feature_names)
    #x_test = pd.DataFrame(x_test, columns=selected_feature_names)

else:
    print("wrong feature selection method...")



#===========================================================
# model building
#===========================================================
# Define classifiers
logreg_classifier = LogisticRegression(max_iter=1000, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
#rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
svm_classifier = SVC(probability=True, random_state=42)
nb_classifier = GaussianNB()


# Train and evaluate models
classifiers = [logreg_classifier, rf_classifier, svm_classifier, nb_classifier]
labels = ['Logistic Regression', 'Random Forest', 'SVM', 'Naive Bayes']
fprs = []
tprs = []
aucs = []
y_probas = []
for classifier, label in zip(classifiers, labels):
    fpr, tpr, auc, y_proba = train_evaluate_model(classifier, x_train, y_train, x_test, y_test)
    fprs.append(fpr)
    tprs.append(tpr)
    aucs.append(auc)
    y_probas.append(y_proba)

# Plot ROC curves for all models
plot_roc_curve(fprs, tprs, aucs, labels)






# Compute metrics for train data
cms_train = []
sensitivities_train = []
specificities_train = []
ppvs_train = []
npvs_train = []
aucs_train = []
for classifier, label in zip(classifiers, labels):
    cm_train, sensitivity_train, specificity_train, ppv_train, npv_train = cross_validate_metrics(classifier, x_train, y_train)
    auc_train = np.mean(cross_val_score(classifier, x_train, y_train, cv=5, scoring='roc_auc'))
    cms_train.append(cm_train)
    sensitivities_train.append(sensitivity_train)
    specificities_train.append(specificity_train)
    ppvs_train.append(ppv_train)
    npvs_train.append(npv_train)
    aucs_train.append(auc_train)

# Compute metrics for test data
cms_test = []
sensitivities_test = []
specificities_test = []
ppvs_test = []
npvs_test = []
aucs_test = []
for classifier, label in zip(classifiers, labels):
    classifier.fit(x_train, y_train)
    y_pred_test = classifier.predict(x_test)
    cm_test, sensitivity_test, specificity_test, ppv_test, npv_test = compute_metrics(y_test, y_pred_test)
    plot_confusion_matrix(cm_test, label)
    auc_test = roc_auc_score(y_test, classifier.predict_proba(x_test)[:, 1])
    cms_test.append(cm_test)
    sensitivities_test.append(sensitivity_test)
    specificities_test.append(specificity_test)
    ppvs_test.append(ppv_test)
    npvs_test.append(npv_test)
    aucs_test.append(auc_test)

# Create and save metrics table
table_data = {
    'Model': labels,
    'AUC Train': ["{:.2f} ({:.2f}-{:.2f})".format(auc, np.percentile(auc_list, 2.5), np.percentile(auc_list, 97.5)) for auc, auc_list in zip(aucs_train, aucs_train)],
    'AUC Test': ["{:.2f} ({:.2f}-{:.2f})".format(auc, np.percentile(auc_list, 2.5), np.percentile(auc_list, 97.5)) for auc, auc_list in zip(aucs_test, aucs_test)],
    'Sensitivity Train': sensitivities_train,
    'Sensitivity Test': sensitivities_test,
    'Specificity Train': specificities_train,
    'Specificity Test': specificities_test,
    'PPV Train': ppvs_train,
    'PPV Test': ppvs_test,
    'NPV Train': npvs_train,
    'NPV Test': npvs_test
}
df_table = pd.DataFrame(table_data)
df_table.to_csv(res_folder + '/' + 'metrics_table.csv', index=False)



## Save models
# dump(logreg_classifier, 'logreg_model.joblib')
# dump(rf_classifier, 'rf_model.joblib')
# dump(svm_classifier, 'svm_model.joblib')
# dump(nb_classifier, 'nb_model.joblib')

# Plot feature importance for Random Forest classifier
plot_feature_importance(rf_classifier, x_train.columns, 'Random Forest')
plot_feature_importance(svm_classifier, x_train.columns, 'SVM')
plot_feature_importance(logreg_classifier, x_train.columns, 'Logistic Regression')
plot_feature_importance(nb_classifier, x_train.columns, 'Naive Bayes')





#==================================

#
# # Calculate AUC scores for each feature
# auc_scores = {}
# for feature in x.columns:
#     auc_scores[feature] = roc_auc_score(y, x[feature])
#
# # Calculate p-values for each feature
# p_values = []
# for feature in x.columns:
#     p_value = ranksums(x[y == 0][feature], x[y == 1][feature])[1]
#     p_values.append(p_value)
#
# # Create DataFrame for AUC scores and p-values
# summary_stats = pd.DataFrame({
#     "feature": x.columns,
#     "AUC": [auc_scores[feature] for feature in x.columns],
#     "p_value": p_values
# })
#
# # Sort DataFrame by AUC values
# #summary_stats.sort_values(by="AUC", ascending=False, inplace=True)
#
# # Save the DataFrame as a CSV file
# #summary_stats.to_csv("summary_stats.csv", index=False)
# with pd.ExcelWriter("summary_stats.xlsx", mode='a') as writer:
#     summary_stats.to_excel(writer, sheet_name=sheetName, index=False)
#
# # Display the DataFrame
# print(summary_stats)
