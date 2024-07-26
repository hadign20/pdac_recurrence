import os
import pandas as pd
import numpy as np
from src.feature_selection.correlation import remove_collinear_features
from src.feature_selection.feature_selection import *
from src.model.model_building import evaluate_models, save_classification_results



#=========================================
# set paths
#=========================================
data_path = r'D:\projects\colonMSI\data'
result_path = r'./results'
img_data_path = os.path.join(data_path, "Segmentations")
excel_file_name = "TumorTexture"
SELECTED_SHEET = "all" #"2_1"
outcome_column = "Outcome"
exclude_columns = ["Case"]
categorical_columns = []


features_file = os.path.join(data_path, excel_file_name + ".xlsx")
results_dir = os.path.join(result_path, excel_file_name)
os.makedirs(results_dir, exist_ok=True)

#=========================================
# set parameters
#=========================================
FEATURE_CORRELATION = True
CORR_THRESH = 0.8

FEATURE_SELECTION = True
FEATURE_SELECTION_METHOD = 'composite' # 'mrmr', 'pvalue', 'auc', 'composite'
min_num_features = 1
max_num_features = 20

MODEL_BUILDING = True
EVALUATION_METHOD = 'cross_validation' # 'train_test_split' or 'cross_validation'
TEST_SIZE = 0.3
CV_FOLDS = 5
HYPERPARAMETER_TUNING = True




#=========================================
def save_excel_sheet(df, filepath, sheetname, index=False):
    # Create file if it does not exist
    if not os.path.exists(filepath):
        df.to_excel(filepath, sheet_name=sheetname, index=index)

    # Otherwise, add a sheet. Overwrite if there exists one with the same name.
    else:
        with pd.ExcelWriter(filepath, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer:
            df.to_excel(writer, sheet_name=sheetname, index=index)
#=========================================


def main():
    xls = pd.ExcelFile(features_file)
    summary_results = []
    best_result = None


    if SELECTED_SHEET == "all":
        for selected_sheet in xls.sheet_names:
            result_dir = os.path.join(results_dir, selected_sheet)
            os.makedirs(result_dir, exist_ok=True)

            df = pd.read_excel(xls, sheet_name=selected_sheet)

            # =========================================
            # Feature selection
            # =========================================
            if FEATURE_CORRELATION:
                print("\n======================================================================")
                print(f"Removing correlated features for sheet {selected_sheet}")
                print("======================================================================")
                df = remove_collinear_features(df, CORR_THRESH)

            if FEATURE_SELECTION:
                print("\n======================================================================")
                print(f"Performing feature analysis for sheet {selected_sheet}")
                print("======================================================================")
                p_values_df = calculate_p_values(df, outcome_column, categorical_columns, exclude_columns)
                auc_values_df = calculate_auc_values(df, outcome_column, categorical_columns, exclude_columns)
                mrmr_df = MRMR_feature_count(df, outcome_column, categorical_columns, exclude_columns, max_num_features, CV_FOLDS)
                composite_df = calculate_feature_scores(p_values_df, auc_values_df, mrmr_df, result_dir)

                save_feature_analysis(p_values_df, auc_values_df, mrmr_df, composite_df, result_dir)

                df_copy = df.copy()

                for num_features in range(min_num_features, max_num_features + 1):
                    print("\n======================================================================")
                    print(f"Selecting {num_features} significant features for sheet {selected_sheet}")
                    print("======================================================================")

                    selected_features = []
                    if FEATURE_SELECTION_METHOD == 'mrmr':
                        selected_features = mrmr_df['Feature'][:num_features].tolist()
                        print(f"{num_features} features were selected by using MRMR method")
                    elif FEATURE_SELECTION_METHOD == 'pvalue':
                        selected_features = p_values_df['Feature'][:num_features].tolist()
                        print(f"{num_features} features were selected by using pvalue method")
                    elif FEATURE_SELECTION_METHOD == 'auc':
                        selected_features = auc_values_df['Feature'][:num_features].tolist()
                        print(f"{num_features} features were selected by using auc method")
                    elif FEATURE_SELECTION_METHOD == 'composite':
                        selected_features = composite_df['Feature'][:num_features].tolist()
                        print(f"{num_features} features were selected by a composite of p_value, AUC, and MRMR method")
                    else:
                        raise ValueError("FEATURE_SELECTION_METHOD is not correct. It should be 'mrmr', 'pvalue', 'auc', or 'composite'")

                    df = df_copy[exclude_columns + selected_features + [outcome_column]]

                    # =========================================
                    # Model building and evaluation
                    # =========================================
                    if MODEL_BUILDING:
                        eval_kwargs = {'test_size': TEST_SIZE,
                                       'random_state': 42} if EVALUATION_METHOD == 'train_test_split' else {'cv_folds': CV_FOLDS}

                        print("\n======================================================================")
                        print(f"Training and evaluating classification models for {num_features} feature(s) in sheet {selected_sheet}")
                        print("======================================================================")
                        X = df.loc[:, ~df.columns.isin(exclude_columns + [outcome_column])]
                        y = df[outcome_column]

                        classification_results = evaluate_models(X, y, method=EVALUATION_METHOD, **eval_kwargs)

                        classification_results_file = os.path.join(result_dir, 'model_evaluation_results.xlsx')
                        save_classification_results(classification_results, classification_results_file, num_features, method=EVALUATION_METHOD)

                        # Record summary results
                        for classifier, result in classification_results.items():
                            result_entry = {
                                'Sheet': selected_sheet,
                                'Num Features': num_features,
                                'Classifier': classifier,
                                'AUC': result['metrics']['roc_auc'],
                                'Sensitivity': result['metrics']['sensitivity'],
                                'Specificity': result['metrics']['specificity'],
                                'PPV': result['metrics']['ppv'],
                                'NPV': result['metrics']['npv']
                            }
                            summary_results.append(result_entry)
                            if best_result is None or result['metrics']['roc_auc'] > best_result['AUC']:
                                best_result = result_entry

        # Save summary results
        summary_df = pd.DataFrame(summary_results)
        summary_file = os.path.join(results_dir, 'summary_results.xlsx')
        with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
            for sheet_name in summary_df['Sheet'].unique():
                sheet_df = summary_df[summary_df['Sheet'] == sheet_name]
                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
            if best_result:
                best_df = pd.DataFrame([best_result])
                best_df.to_excel(writer, sheet_name='Best Result', index=False)

    else:
        result_dir = os.path.join(results_dir, SELECTED_SHEET)
        os.makedirs(result_dir, exist_ok=True)

        df = pd.read_excel(xls, sheet_name=SELECTED_SHEET)

        # =========================================
        # Feature selection
        # =========================================
        if FEATURE_CORRELATION:
            print("\n======================================================================")
            print(f"Removing correlated features for sheet {SELECTED_SHEET}")
            print("======================================================================")
            df = remove_collinear_features(df, CORR_THRESH)

        if FEATURE_SELECTION:
            print("\n======================================================================")
            print(f"Performing feature analysis for sheet {SELECTED_SHEET}")
            print("======================================================================")
            p_values_df = calculate_p_values(df, outcome_column, categorical_columns, exclude_columns)
            auc_values_df = calculate_auc_values(df, outcome_column, categorical_columns, exclude_columns)
            mrmr_df = MRMR_feature_count(df, outcome_column, categorical_columns, exclude_columns, max_num_features,
                                         CV_FOLDS)
            composite_df = calculate_feature_scores(p_values_df, auc_values_df, mrmr_df, result_dir)

            save_feature_analysis(p_values_df, auc_values_df, mrmr_df, composite_df, result_dir)

            df_copy = df.copy()

            for num_features in range(min_num_features, max_num_features + 1):
                print("\n======================================================================")
                print(f"Selecting {num_features} significant features for sheet {SELECTED_SHEET}")
                print("======================================================================")

                selected_features = []
                if FEATURE_SELECTION_METHOD == 'mrmr':
                    selected_features = mrmr_df['Feature'][:num_features].tolist()
                    print(f"{num_features} features were selected by using MRMR method")
                elif FEATURE_SELECTION_METHOD == 'pvalue':
                    selected_features = p_values_df['Feature'][:num_features].tolist()
                    print(f"{num_features} features were selected by using pvalue method")
                elif FEATURE_SELECTION_METHOD == 'auc':
                    selected_features = auc_values_df['Feature'][:num_features].tolist()
                    print(f"{num_features} features were selected by using auc method")
                elif FEATURE_SELECTION_METHOD == 'composite':
                    selected_features = composite_df['Feature'][:num_features].tolist()
                    print(f"{num_features} features were selected by a composite of p_value, AUC, and MRMR method")
                else:
                    raise ValueError(
                        "FEATURE_SELECTION_METHOD is not correct. It should be 'mrmr', 'pvalue', 'auc', or 'composite'")

                df = df_copy[exclude_columns + selected_features + [outcome_column]]

                # =========================================
                # Model building and evaluation
                # =========================================
                if MODEL_BUILDING:
                    eval_kwargs = {'test_size': TEST_SIZE,
                                   'random_state': 42} if EVALUATION_METHOD == 'train_test_split' else {
                        'cv_folds': CV_FOLDS}

                    print("\n======================================================================")
                    print(f"Training and evaluating classification models for {num_features} feature(s) in sheet {SELECTED_SHEET}")
                    print("======================================================================")
                    X = df.loc[:, ~df.columns.isin(exclude_columns + [outcome_column])]
                    y = df[outcome_column]

                    classification_results = evaluate_models(X, y, method=EVALUATION_METHOD, **eval_kwargs)

                    classification_results_file = os.path.join(result_dir, 'model_evaluation_results.xlsx')
                    save_classification_results(classification_results, classification_results_file, num_features,
                                                method=EVALUATION_METHOD)

                    # Record summary results
                    for classifier, result in classification_results.items():
                        result_entry = {
                            'Sheet': SELECTED_SHEET,
                            'Num Features': num_features,
                            'Classifier': classifier,
                            'AUC': result['metrics']['roc_auc'],
                            'Sensitivity': result['metrics']['sensitivity'],
                            'Specificity': result['metrics']['specificity'],
                            'PPV': result['metrics']['ppv'],
                            'NPV': result['metrics']['npv']
                        }
                        summary_results.append(result_entry)
                        if best_result is None or result['metrics']['roc_auc'] > best_result['AUC']:
                            best_result = result_entry

        # Save summary results
    summary_df = pd.DataFrame(summary_results)
    summary_file = os.path.join(results_dir, 'summary_results.xlsx')
    with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
        for sheet_name in summary_df['Sheet'].unique():
            sheet_df = summary_df[summary_df['Sheet'] == sheet_name]
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
        if best_result:
            best_df = pd.DataFrame([best_result])
            best_df.to_excel(writer, sheet_name='Best Result', index=False)


if __name__ == '__main__':
    main()

