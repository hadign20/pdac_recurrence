# import os
#
# import numpy as np
# import pandas as pd
# from scipy.stats import ttest_ind, fisher_exact, wilcoxon, chi2_contingency, mannwhitneyu, ranksums
# from sklearn.metrics import roc_curve, roc_auc_score, auc
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
# import seaborn as sns
# from mrmr import mrmr_classif
# from sklearn.model_selection import KFold
# from collections import defaultdict
# from typing import List, Optional
# from openpyxl import load_workbook
#
# def calculate_p_values(df: pd.DataFrame,
#                        outcome_column: str,
#                        categorical_columns: List[str] = [],
#                        exclude_columns: List[str] = [],
#                        test_numeric: Optional[str] = 'wilcox',
#                        test_categorical: Optional[str] = 'fisher') -> pd.DataFrame:
#     """
#     Calculate p-values for each feature in the dataframe compared to the outcome variable.
#
#     :param df: DataFrame containing features and the outcome variable.
#     :param outcome_column: The name of the outcome column.
#     :param categorical_columns: List of names of categorical feature columns.
#     :param exclude_columns: List of columns to exclude from the analysis.
#     :param test_numeric: Statistical test to use for numeric features ('ttest' or 'wilcox').
#     :param test_categorical: Statistical test to use for categorical features ('fisher' or 'chi2').
#     :return: DataFrame with features and their corresponding p-values.
#     """
#     p_values = {}
#
#     for column in df.columns:
#         if column in exclude_columns or column == outcome_column: continue
#
#         if column in categorical_columns:
#             if test_categorical == 'fisher':
#                 try:
#                     contingency_table = pd.crosstab(df[column], df[outcome_column])
#                     _, p_value = fisher_exact(contingency_table)
#                 except ValueError:
#                     p_value = np.nan
#             elif test_categorical == 'chi2':
#                 try:
#                     contingency_table = pd.crosstab(df[column], df[outcome_column])
#                     _, p_value = chi2_contingency(contingency_table)
#                 except ValueError:
#                     p_value = np.nan
#             else:
#                 raise ValueError("Invalid test for categorical features. Choose 'fisher' or 'chi2'.")
#
#         elif pd.api.types.is_numeric_dtype(df[column]):
#             if test_numeric == 'ttest':
#                 _, p_value = ttest_ind(df[column], df[outcome_column])
#             elif test_numeric == 'wilcox':
#                 try:
#                     group_0 = df[df[outcome_column] == 0][column]
#                     group_1 = df[df[outcome_column] == 1][column]
#                     _, p_value = mannwhitneyu(group_0, group_1)
#                     #_, p_value = wilcoxon(df[column], df[outcome_column])
#                 except ValueError:
#                     p_value = np.nan
#             else:
#                 raise ValueError("Invalid test for numeric features. Choose 'ttest' or 'wilcoxon'.")
#         else:
#             raise ValueError(f"Column {column} not specified in categorical or numerical columns list.")
#
#         p_values[column] = p_value
#
#     p_values_df = pd.DataFrame(list(p_values.items()), columns=['Feature', 'P_Value'])
#     p_values_df = p_values_df.sort_values(by=['P_Value'], ascending=True)
#     return p_values_df
#
#
# def calculate_auc_values(df: pd.DataFrame,
#                        outcome_column: str,
#                        categorical_columns: List[str] = [],
#                        exclude_columns: List[str] = [] ) -> pd.DataFrame:
#     """
#     Calculate auc-values for each feature in the dataframe compared to the outcome variable.
#
#     :param df: DataFrame containing features and the outcome variable.
#     :param outcome_column: The name of the outcome column.
#     :param categorical_columns: List of names of categorical feature columns.
#     :param exclude_columns: List of columns to exclude from the analysis.
#     :return: DataFrame with features and their corresponding auc-values.
#     """
#     auc_values = {}
#     scaler = MinMaxScaler()
#     #df[outcome_column] = pd.factorize(df[outcome_column])[0]
#     for column in df.columns:
#         if column == outcome_column or column in exclude_columns: continue
#
#         if column in categorical_columns:
#             df[column] = pd.factorize(df[column])[0]
#
#         feature_values = df[column].values.reshape(-1, 1)
#         normalized_feature_values = scaler.fit_transform(feature_values).flatten()
#
#         if len(df[outcome_column].unique()) == 2:
#             try:
#                 fpr, tpr, _ = roc_curve(df[outcome_column], normalized_feature_values)
#                 roc_auc = auc(fpr, tpr)
#                 auc_values[column] = roc_auc if roc_auc >= 0.5 else (1 - roc_auc)
#             except ValueError:
#                 auc_values[column] = np.nan
#         else:
#             auc_values[column] = np.nan
#
#     auc_values_df = pd.DataFrame(list(auc_values.items()), columns=['Feature', 'AUC'])
#     auc_values_df = auc_values_df.sort_values(by=['AUC'], ascending=False)
#     return auc_values_df
#
#
#
#
# def MRMR_feature_count(df: pd.DataFrame,
#                            outcome_column: str,
#                            categorical_columns: List[str] = [],
#                            exclude_columns: List[str] = [],
#                            num_features: Optional[int] = 10,
#                            CV_folds: Optional[int] = 20) -> pd.DataFrame:
#     """
#     Select best features defined by cross validation on MRMR method.
#
#     :param df: DataFrame containing features and the outcome variable.
#     :param outcome_column: The name of the outcome column.
#     :param exclude_columns: List of columns to exclude from the analysis.
#     :return: DataFrame with MRMR-selected features.
#     """
#     print("Selecting best features defined by cross validation on MRMR method.")
#     x = df.loc[:, ~df.columns.isin(exclude_columns + [outcome_column])]
#     y = df[outcome_column]
#
#     kf = KFold(n_splits=CV_folds)
#     selected_feature_count = defaultdict(int)
#     for feature in x.columns:
#         selected_feature_count[feature] = 0
#
#     for train_index, val_index in kf.split(x):
#         x_train_fold, x_val_fold = x.iloc[train_index], x.iloc[val_index]
#         y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
#         selected_features_fold = mrmr_classif(X = x_train_fold, y = y_train_fold, K = num_features)
#         for feature in selected_features_fold:
#             selected_feature_count[feature] += 1
#
#     mrmr_count_df = pd.DataFrame(list(selected_feature_count.items()), columns=['Feature', 'MRMR_Count'])
#     mrmr_count_df = mrmr_count_df.sort_values(by=['MRMR_Count'], ascending=False)
#     return mrmr_count_df
#
#
#
# def calculate_feature_scores(p_values_df: pd.DataFrame,
#                           auc_values_df: pd.DataFrame,
#                           mrmr_count_df: pd.DataFrame,
#                           results_dir: str):
#     """
#     factor in p-value, AUC, and MRMR count simultaneously,
#     to create a composite score that combines these metrics.
#     This composite score can then be used to rank the features.
#     :param p_values_df: DF of feature p-values.
#     :param auc_values_df: DF of feature AUC values.
#     :param mrmr_count_df: DF of selected features by MRMR.
#     :param results_dir: Path to reulsts directory.
#     :return: DataFrame with selected features.
#     """
#     # normalize each metric
#     normalized_df = pd.DataFrame()
#     normalized_df['Feature'] = p_values_df['Feature']
#     normalized_df['Normalized_P_Value'] = (p_values_df['P_Value'] - p_values_df['P_Value'].min()) / (
#             p_values_df['P_Value'].max() - p_values_df['P_Value'].min())
#     normalized_df['Normalized_AUC'] = (auc_values_df['AUC'] - auc_values_df['AUC'].min()) / (
#             auc_values_df['AUC'].max() - auc_values_df['AUC'].min())
#     normalized_df['Normalized_MRMR_Count'] = (mrmr_count_df['MRMR_Count'] - mrmr_count_df['MRMR_Count'].min()) / (
#             mrmr_count_df['MRMR_Count'].max() - mrmr_count_df['MRMR_Count'].min())
#
#     # Assign weights to each metric
#     w_p_value = 0.2
#     w_auc = 0.4
#     w_mrmr_count = 0.4
#
#     # Calculate composite score
#     normalized_df['Composite_Score'] = (w_p_value * (1 - normalized_df['Normalized_P_Value']) +
#                                         w_auc * normalized_df['Normalized_AUC'] +
#                                         w_mrmr_count * normalized_df['Normalized_MRMR_Count'])
#
#     # Sort features based on the composite score
#     normalized_df = normalized_df.sort_values(by='Composite_Score', ascending=False)
#
#     # Plot the composite score for each feature
#     plt.figure(figsize=(12,8))
#     plt.barh(normalized_df['Feature'], normalized_df['Composite_Score'], color='skyblue')
#     plt.xlabel('Composite Score')
#     plt.ylabel('Features')
#     plt.title('Feature Importance based on Composite Score')
#     plt.gca().invert_yaxis()
#
#     #output_dir = os.path.join(results_dir, "feature_analysis")
#     #os.makedirs(output_dir, exist_ok=True)
#     plt.savefig(os.path.join(results_dir, 'feature_composite_score.png'))
#
#     return normalized_df
#
#
# def save_feature_analysis(p_values_df: pd.DataFrame,
#                           auc_values_df: pd.DataFrame,
#                           mrmr_count_df: pd.DataFrame,
#                           composite_df: pd.DataFrame,
#                           results_dir: str):
#     """
#     Save the resutls of feature analysis to the output dir.
#
#     :param p_values_df: DF of feature p-values.
#     :param auc_values_df: DF of feature AUC values.
#     :param mrmr_count_df: DF of selected features by MRMR.
#     :param composite_df: DF of selected features by combination.
#     :param results_dir: Path to reulsts directory.
#     """
#     analysis_df = p_values_df.merge(auc_values_df, on='Feature').merge(mrmr_count_df, on='Feature').merge(composite_df, on='Feature')
#     analysis_df = analysis_df.sort_values(by=['Composite_Score', 'AUC', 'P_Value', 'MRMR_Count'], ascending=[False, False, True, False])
#
#     #output_dir = os.path.join(results_dir, "feature_analysis")
#     #os.makedirs(output_dir, exist_ok=True)
#     output_file = os.path.join(results_dir, 'feature_analysis.xlsx')
#     analysis_df.to_excel(output_file, index=False)
#
#
#




import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, fisher_exact, wilcoxon, chi2_contingency, mannwhitneyu, ranksums
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from mrmr import mrmr_classif
from sklearn.model_selection import KFold
from collections import defaultdict
from typing import List, Optional
from openpyxl import load_workbook
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def calculate_p_values(df: pd.DataFrame,
                       outcome_column: str,
                       categorical_columns: List[str] = [],
                       exclude_columns: List[str] = [],
                       test_numeric: Optional[str] = 'wilcox',
                       test_categorical: Optional[str] = 'fisher') -> pd.DataFrame:
    """
    Calculate p-values for each feature in the dataframe compared to the outcome variable.

    :param df: DataFrame containing features and the outcome variable.
    :param outcome_column: The name of the outcome column.
    :param categorical_columns: List of names of categorical feature columns.
    :param exclude_columns: List of columns to exclude from the analysis.
    :param test_numeric: Statistical test to use for numeric features ('ttest' or 'wilcox').
    :param test_categorical: Statistical test to use for categorical features ('fisher' or 'chi2').
    :return: DataFrame with features and their corresponding p-values.
    """
    p_values = {}

    for column in df.columns:
        if column in exclude_columns or column == outcome_column: continue

        if column in categorical_columns:
            if test_categorical == 'fisher':
                try:
                    contingency_table = pd.crosstab(df[column], df[outcome_column])
                    _, p_value = fisher_exact(contingency_table)
                except ValueError:
                    p_value = np.nan
            elif test_categorical == 'chi2':
                try:
                    contingency_table = pd.crosstab(df[column], df[outcome_column])
                    _, p_value = chi2_contingency(contingency_table)
                except ValueError:
                    p_value = np.nan
            else:
                raise ValueError("Invalid test for categorical features. Choose 'fisher' or 'chi2'.")

        elif pd.api.types.is_numeric_dtype(df[column]):
            if test_numeric == 'ttest':
                _, p_value = ttest_ind(df[column], df[outcome_column])
            elif test_numeric == 'wilcox':
                try:
                    group_0 = df[df[outcome_column] == 0][column]
                    group_1 = df[df[outcome_column] == 1][column]
                    _, p_value = mannwhitneyu(group_0, group_1)
                    #_, p_value = wilcoxon(df[column], df[outcome_column])
                except ValueError:
                    p_value = np.nan
            else:
                raise ValueError("Invalid test for numeric features. Choose 'ttest' or 'wilcoxon'.")
        else:
            raise ValueError(f"Column {column} not specified in categorical or numerical columns list.")

        p_values[column] = p_value

    p_values_df = pd.DataFrame(list(p_values.items()), columns=['Feature', 'P_Value'])
    p_values_df = p_values_df.sort_values(by=['P_Value'], ascending=True)
    return p_values_df


def calculate_p_values_CV(df: pd.DataFrame,
                          outcome_column: str,
                          categorical_columns: List[str] = [],
                          exclude_columns: List[str] = [],
                          test_numeric: Optional[str] = 'wilcox',
                          test_categorical: Optional[str] = 'fisher',
                          cv_folds: int = 5) -> pd.DataFrame:
    """
    Calculate p-values for each feature in the dataframe compared to the outcome variable using cross-validation.

    :param df: DataFrame containing features and the outcome variable.
    :param outcome_column: The name of the outcome column.
    :param categorical_columns: List of names of categorical feature columns.
    :param exclude_columns: List of columns to exclude from the analysis.
    :param test_numeric: Statistical test to use for numeric features ('ttest' or 'wilcox').
    :param test_categorical: Statistical test to use for categorical features ('fisher' or 'chi2').
    :param cv_folds: Number of cross-validation folds.
    :return: DataFrame with features and their average p-values across the CV folds.
    """
    print("Selecting best features defined by cross-validation with the p-value method.")

    kf = KFold(n_splits=cv_folds)
    feature_pvalue_avg = defaultdict(float)
    p_values_count = defaultdict(int)

    for train_index, val_index in kf.split(df):
        train_fold, val_fold = df.iloc[train_index], df.iloc[val_index]
        p_values_fold_df = calculate_p_values(train_fold, outcome_column, categorical_columns, exclude_columns,
                                              test_numeric, test_categorical)

        for _, row in p_values_fold_df.iterrows():
            feature = row['Feature']
            p_value = row['P_Value']
            feature_pvalue_avg[feature] += p_value
            p_values_count[feature] += 1

    # Calculate average p-values
    p_values_avg = {feature: feature_pvalue_avg[feature] / p_values_count[feature] for feature in feature_pvalue_avg}

    p_values_avg_df = pd.DataFrame(list(p_values_avg.items()), columns=['Feature', 'P_Value'])
    p_values_avg_df = p_values_avg_df.sort_values(by=['P_Value'], ascending=True)
    return p_values_avg_df




def calculate_auc_values(df: pd.DataFrame,
                       outcome_column: str,
                       categorical_columns: List[str] = [],
                       exclude_columns: List[str] = [] ) -> pd.DataFrame:
    """
    Calculate auc-values for each feature in the dataframe compared to the outcome variable.

    :param df: DataFrame containing features and the outcome variable.
    :param outcome_column: The name of the outcome column.
    :param categorical_columns: List of names of categorical feature columns.
    :param exclude_columns: List of columns to exclude from the analysis.
    :return: DataFrame with features and their corresponding auc-values.
    """
    auc_values = {}
    scaler = MinMaxScaler()
    #df[outcome_column] = pd.factorize(df[outcome_column])[0]
    for column in df.columns:
        if column == outcome_column or column in exclude_columns: continue

        if column in categorical_columns:
            df[column] = pd.factorize(df[column])[0]

        feature_values = df[column].values.reshape(-1, 1)
        normalized_feature_values = scaler.fit_transform(feature_values).flatten()

        if len(df[outcome_column].unique()) == 2:
            try:
                fpr, tpr, _ = roc_curve(df[outcome_column], normalized_feature_values)
                roc_auc = auc(fpr, tpr)
                auc_values[column] = roc_auc
            except ValueError:
                auc_values[column] = np.nan
        else:
            auc_values[column] = np.nan

    auc_values_df = pd.DataFrame(list(auc_values.items()), columns=['Feature', 'AUC'])
    auc_values_df = auc_values_df.sort_values(by=['AUC'], ascending=False)
    return auc_values_df






def calculate_auc_values_CV(df: pd.DataFrame,
                         outcome_column: str,
                         categorical_columns: List[str] = [],
                         exclude_columns: List[str] = [],
                         cv_folds: int = 5) -> pd.DataFrame:
    """
    Calculate cross-validated AUC values for each feature in the dataframe compared to the outcome variable.

    :param df: DataFrame containing features and the outcome variable.
    :param outcome_column: The name of the outcome column.
    :param categorical_columns: List of names of categorical feature columns.
    :param exclude_columns: List of columns to exclude from the analysis.
    :param cv_folds: Number of cross-validation folds.
    :return: DataFrame with features and their corresponding cross-validated AUC values.
    """
    print("Selecting best features defined by cross validation with AUC method.")
    auc_values = {}
    scaler = MinMaxScaler()
    y = df[outcome_column].values

    for column in df.columns:
        if column == outcome_column or column in exclude_columns:
            continue

        if column in categorical_columns:
            df[column] = pd.factorize(df[column])[0]

        feature_values = df[column].values.reshape(-1, 1)
        normalized_feature_values = scaler.fit_transform(feature_values).flatten().reshape(-1, 1)

        if len(np.unique(y)) == 2:
            model = LogisticRegression(solver='liblinear')
            cv = StratifiedKFold(n_splits=cv_folds)
            try:
                auc_scores = cross_val_score(model, normalized_feature_values, y, cv=cv, scoring='roc_auc')
                auc_values[column] = auc_scores.mean()
            except ValueError:
                auc_values[column] = np.nan
        else:
            auc_values[column] = np.nan

    auc_values_df = pd.DataFrame(list(auc_values.items()), columns=['Feature', 'AUC'])
    auc_values_df = auc_values_df.sort_values(by=['AUC'], ascending=False)
    return auc_values_df




def MRMR_feature_count(df: pd.DataFrame,
                           outcome_column: str,
                           categorical_columns: List[str] = [],
                           exclude_columns: List[str] = [],
                           num_features: Optional[int] = 10,
                           CV_folds: Optional[int] = 20) -> pd.DataFrame:
    """
    Select best features defined by cross validation on MRMR method.

    :param df: DataFrame containing features and the outcome variable.
    :param outcome_column: The name of the outcome column.
    :param exclude_columns: List of columns to exclude from the analysis.
    :return: DataFrame with MRMR-selected features.
    """
    print("Selecting best features defined by cross validation on MRMR method.")
    x = df.loc[:, ~df.columns.isin(exclude_columns + [outcome_column])]
    y = df[outcome_column]

    kf = KFold(n_splits=CV_folds)
    selected_feature_count = defaultdict(int)
    for feature in x.columns:
        selected_feature_count[feature] = 0

    for train_index, val_index in kf.split(x):
        x_train_fold, x_val_fold = x.iloc[train_index], x.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        selected_features_fold = mrmr_classif(X = x_train_fold, y = y_train_fold, K = num_features)
        for feature in selected_features_fold:
            selected_feature_count[feature] += 1

    mrmr_count_df = pd.DataFrame(list(selected_feature_count.items()), columns=['Feature', 'MRMR_Count'])
    mrmr_count_df = mrmr_count_df.sort_values(by=['MRMR_Count'], ascending=False)
    return mrmr_count_df


def lasso_feature_selection(df: pd.DataFrame, outcome_column: str, exclude_columns: List[str] = [],
                            alphas: List[float] = [0.1, 1.0, 10.0]) -> pd.DataFrame:
    x = df.loc[:, ~df.columns.isin(exclude_columns + [outcome_column])]
    y = df[outcome_column]

    # Normalize the features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Perform LassoCV with a range of alphas
    lasso = LassoCV(alphas=alphas, cv=5).fit(x_scaled, y)

    # Select features where coefficients are not zero
    selected_features = x.columns[lasso.coef_ != 0]
    lasso_df = pd.DataFrame(list(selected_features), columns=['Feature'])
    lasso_df['Lasso_Coefficient'] = lasso.coef_[lasso.coef_ != 0]
    lasso_df = lasso_df.sort_values(by='Lasso_Coefficient', ascending=False)

    return lasso_df


def pca_feature_selection(df: pd.DataFrame, outcome_column: str, exclude_columns: List[str] = [], n_components: int = 2) -> pd.DataFrame:
    print("Selecting best features defined by PCA method.")
    exclude_columns = [col for col in exclude_columns if col not in [df.columns[0], outcome_column]]

    # Separate features and outcome
    x = df.loc[:, ~df.columns.isin(exclude_columns + [outcome_column])]
    y = df[outcome_column]

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(x.drop(columns=[x.columns[0]]))  # Dropping CaseNo for PCA

    # Create a DataFrame with the PCA components
    pca_df = pd.DataFrame(pca_features, columns=[f'PC{i + 1}' for i in range(n_components)])

    # Add CaseNo and outcome_column back to the DataFrame
    pca_df.insert(0, x.columns[0], x[x.columns[0]])
    pca_df[outcome_column] = y.values

    return pca_df



def calculate_feature_scores(p_values_df: pd.DataFrame,
                          auc_values_df: pd.DataFrame,
                          mrmr_count_df: pd.DataFrame,
                          lasso_values_df: pd.DataFrame,
                          results_dir: str):
    """
    factor in p-value, AUC, and MRMR count simultaneously,
    to create a composite score that combines these metrics.
    This composite score can then be used to rank the features.
    :param p_values_df: DF of feature p-values.
    :param auc_values_df: DF of feature AUC values.
    :param mrmr_count_df: DF of selected features by MRMR.
    :param results_dir: Path to reulsts directory.
    :return: DataFrame with selected features.
    """
    # normalize each metric
    normalized_df = pd.DataFrame()
    normalized_df['Feature'] = p_values_df['Feature']
    normalized_df['Normalized_P_Value'] = (p_values_df['P_Value'] - p_values_df['P_Value'].min()) / (
            p_values_df['P_Value'].max() - p_values_df['P_Value'].min())
    normalized_df['Normalized_AUC'] = (auc_values_df['AUC'] - auc_values_df['AUC'].min()) / (
            auc_values_df['AUC'].max() - auc_values_df['AUC'].min())
    normalized_df['Normalized_MRMR_Count'] = (mrmr_count_df['MRMR_Count'] - mrmr_count_df['MRMR_Count'].min()) / (
            mrmr_count_df['MRMR_Count'].max() - mrmr_count_df['MRMR_Count'].min())
    normalized_df['Normalized_Lasso'] = (lasso_values_df['Lasso_Coefficient'] - lasso_values_df['Lasso_Coefficient'].min()) / (
            lasso_values_df['Lasso_Coefficient'].max() - lasso_values_df['Lasso_Coefficient'].min())

    # Assign weights to each metric
    w_p_value = 0.4
    w_auc = 0.2
    w_mrmr_count = 0.4
    w_lasso = 0.0

    # Calculate composite score
    # normalized_df['Composite_Score'] = (w_p_value * (1 - normalized_df['Normalized_P_Value']) +
    #                                     w_auc * normalized_df['Normalized_AUC'] +
    #                                     w_mrmr_count * normalized_df['Normalized_MRMR_Count'] +
    #                                     w_lasso * normalized_df['Normalized_Lasso'])

    normalized_df['Composite_Score'] = (w_p_value * (1 - normalized_df['Normalized_P_Value']) +
                                        w_auc * normalized_df['Normalized_AUC'] +
                                        w_mrmr_count * normalized_df['Normalized_MRMR_Count'])

    # Sort features based on the composite score
    normalized_df = normalized_df.sort_values(by='Composite_Score', ascending=False)

    # Plot the composite score for each feature
    plt.figure(figsize=(12,8))
    plt.barh(normalized_df['Feature'], normalized_df['Composite_Score'], color='skyblue')
    plt.xlabel('Composite Score')
    plt.ylabel('Features')
    plt.title('Feature Importance based on Composite Score')
    plt.gca().invert_yaxis()

    #output_dir = os.path.join(results_dir, "feature_analysis")
    #os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'feature_composite_score.png'))

    return normalized_df


def save_feature_analysis(p_values_df: pd.DataFrame,
                          auc_values_df: pd.DataFrame,
                          mrmr_count_df: pd.DataFrame,
                          lasso_values_df: pd.DataFrame,
                          composite_df: pd.DataFrame,
                          results_dir: str):
    """
    Save the resutls of feature analysis to the output dir.

    :param p_values_df: DF of feature p-values.
    :param auc_values_df: DF of feature AUC values.
    :param mrmr_count_df: DF of selected features by MRMR.
    :param composite_df: DF of selected features by combination.
    :param results_dir: Path to reulsts directory.
    """
    #analysis_df = p_values_df.merge(auc_values_df, on='Feature').merge(mrmr_count_df, on='Feature').merge(composite_df, on='Feature').merge(lasso_values_df, on='Feature')
    #analysis_df = analysis_df.sort_values(by=['Composite_Score', 'AUC', 'P_Value', 'MRMR_Count', 'Lasso_Coefficient'], ascending=[False, False, True, False, False])

    analysis_df = p_values_df.merge(auc_values_df, on='Feature').merge(mrmr_count_df, on='Feature').merge(composite_df,
                                                                                                          on='Feature')
    analysis_df = analysis_df.sort_values(by=['Composite_Score', 'AUC', 'P_Value', 'MRMR_Count'],
                                          ascending=[False, False, True, False])

    #output_dir = os.path.join(results_dir, "feature_analysis")
    #os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(results_dir, 'feature_analysis.xlsx')
    analysis_df.to_excel(output_file, index=False)



