import os
import pandas as pd
import numpy as np
from src.feature_selection.correlation import remove_collinear_features
from src.feature_selection.feature_selection import *
from src.model.model_building import evaluate_models, save_classification_results
from src.feature_extraction.radiomics_features import extract_radiomics_features
from src.feature_extraction.deep_features import extract_deep_features
import faulthandler



#=========================================
# set paths
#=========================================
data_path = r'./data'
result_path = r'./results'
img_data_path = r'D:\projects\PDAC Recurrence\data\scans'

params_path_radiomics = r'./src/feature_extraction/CT.yaml'
feature_file_names_radiomics = ["JayaTexturePancreasMedian", "JayaTextureLiverMedian"]
SELECTED_SHEET_RADIOMICS = "all" #"2_1"
outcome_column = "EarlyRecurrence"
exclude_columns = ["CaseNo"]
categorical_columns = []

#================================
params_path_dl = r'./src/feature_extraction/CT.yaml'
feature_file_names_dl = ["HadiDLLiver", "HadiDLPancreas"]
SELECTED_SHEET_DL = "all" #"2_1"



#=========================================
# set parameters
#=========================================
RADIOMICS = False

FEATURE_EXTRACTION_RADIOMICS = False

FEATURE_CORRELATION_RADIOMICS = True
CORR_THRESH_RADIOMICS = 0.8

FEATURE_SELECTION_RADIOMICS = True
FEATURE_SELECTION_METHOD_RADIOMICS = 'auc' # 'mrmr', 'pvalue', 'auc', 'composite'
min_num_features_radiomics = 1
max_num_features_radiomics = 20

MODEL_BUILDING_RADIOMICS = True
EVALUATION_METHOD_RADIOMICS = 'cross_validation' # 'train_test_split' or 'cross_validation'
TEST_SIZE_RADIOMICS = 0.3
CV_FOLDS_RADIOMICS = 5
HYPERPARAMETER_TUNING_RADIOMICS = False

#================================
DEEP_LEANING = True
FEATURE_EXTRACTION_DEEP = False
dl_model_name = 'resnet50'  # ('vgg16', 'densenet121')

FEATURE_CORRELATION_DL = True
CORR_THRESH_DL = 0.8

FEATURE_SELECTION_DL = True
FEATURE_SELECTION_METHOD_DL = 'auc' # 'mrmr', 'pvalue', 'auc', 'composite'
min_num_features_dl = 1
max_num_features_dl = 20

MODEL_BUILDING_DL = True
EVALUATION_METHOD_DL = 'cross_validation' # 'train_test_split' or 'cross_validation'
TEST_SIZE_DL = 0.3
CV_FOLDS_DL = 5
HYPERPARAMETER_TUNING_DL = False




#=========================================
# functions
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
# Main
#=========================================
def main():
    if RADIOMICS:
        # Radiomics Feature Extraction
        if FEATURE_EXTRACTION_RADIOMICS:
            cases = os.listdir(img_data_path)
            liver_features = []
            panc_features = []

            for case in cases:
                print(f"Processing {case}...")
                try:
                    case_folder = os.path.join(img_data_path, case)
                    nifti_files = os.listdir(case_folder)
                    ct_file = [f for f in nifti_files if f.startswith("DICOM") and f.endswith('nii')][0]
                    liver_file = [f for f in nifti_files if f.startswith("liver")][0]
                    panc_file = [f for f in nifti_files if f.startswith("pancreas")][0]
                    ct_path = os.path.join(case_folder, ct_file)
                    liver_path = os.path.join(case_folder, liver_file)
                    panc_path = os.path.join(case_folder, panc_file)

                    if not os.path.exists(liver_path):
                        print(f"file {liver_path} don't exist..!")
                        continue
                    elif not os.path.exists(panc_path):
                        print(f"file {panc_path} don't exist..!")
                        continue
                    elif not os.path.exists(ct_path):
                        print(f"file {ct_path} don't exist..!")
                        continue
                    else:
                        try:
                            l_features = extract_radiomics_features(ct_path, liver_path, params_path_radiomics)
                            liver_features_row = {'CaseNo': case}
                            liver_features_row.update(l_features)
                            liver_features.append(liver_features_row)

                            p_features = extract_radiomics_features(ct_path, panc_path, params_path_radiomics)
                            panc_features_row = {'CaseNo': case}
                            panc_features_row.update(p_features)
                            panc_features.append(panc_features_row)
                        except Exception as e:
                            print(f"Case {case} couldn't be processed because: ", e)

                except Exception as e1:
                    print(f"Case {case} couldn't be processed because: ", e1)

            df1 = pd.DataFrame(liver_features)
            df2 = pd.DataFrame(panc_features)
            writer = pd.ExcelWriter('radiomics_features.xlsx', engine='xlsxwriter')
            df1.to_excel(writer, sheet_name='liver_radiomics_features', index=False)
            df2.to_excel(writer, sheet_name='pancreas_radiomics_features', index=False)
            writer.close()


        # Radiomics feature selection and model building
        for excel_file_name in feature_file_names_radiomics:
            features_file = os.path.join(data_path, excel_file_name + ".xlsx")
            results_dir = os.path.join(result_path, excel_file_name)
            os.makedirs(results_dir, exist_ok=True)


            xls = pd.ExcelFile(features_file)
            summary_results = []
            best_result = None

            selected_sheets = []
            if SELECTED_SHEET_RADIOMICS == "all": selected_sheets = xls.sheet_names
            else: selected_sheets.append(SELECTED_SHEET_RADIOMICS)

            for sheet in selected_sheets:
                result_dir = os.path.join(results_dir, sheet)
                os.makedirs(result_dir, exist_ok=True)

                df = pd.read_excel(xls, sheet_name=sheet)

                # Radiomics feature selection
                if FEATURE_CORRELATION_RADIOMICS:
                    print("\n======================================================================")
                    print(f"Removing correlated features for sheet {sheet}")
                    print("======================================================================")
                    df = remove_collinear_features(df, CORR_THRESH_RADIOMICS)

                if FEATURE_SELECTION_RADIOMICS:
                    print("\n======================================================================")
                    print(f"Performing feature analysis for sheet {sheet}")
                    print("======================================================================")
                    p_values_df = calculate_p_values(df, outcome_column, categorical_columns, exclude_columns)
                    auc_values_df = calculate_auc_values_CV(df, outcome_column, categorical_columns, exclude_columns)
                    mrmr_df = MRMR_feature_count(df, outcome_column, categorical_columns, exclude_columns, max_num_features_radiomics, CV_FOLDS_RADIOMICS)
                    composite_df = calculate_feature_scores(p_values_df, auc_values_df, mrmr_df, result_dir)

                    save_feature_analysis(p_values_df, auc_values_df, mrmr_df, composite_df, result_dir)

                    df_copy = df.copy()

                    for num_features in range(min_num_features_radiomics, max_num_features_radiomics + 1):
                        print("\n======================================================================")
                        print(f"Selecting {num_features} significant features for sheet {sheet}")
                        print("======================================================================")

                        selected_features = []
                        if FEATURE_SELECTION_METHOD_RADIOMICS == 'mrmr':
                            selected_features = mrmr_df['Feature'][:num_features].tolist()
                            print(f"{num_features} features were selected by using MRMR method")
                        elif FEATURE_SELECTION_METHOD_RADIOMICS == 'pvalue':
                            selected_features = p_values_df['Feature'][:num_features].tolist()
                            print(f"{num_features} features were selected by using pvalue method")
                        elif FEATURE_SELECTION_METHOD_RADIOMICS == 'auc':
                            selected_features = auc_values_df['Feature'][:num_features].tolist()
                            print(f"{num_features} features were selected by using auc method")
                        elif FEATURE_SELECTION_METHOD_RADIOMICS == 'composite':
                            selected_features = composite_df['Feature'][:num_features].tolist()
                            print(f"{num_features} features were selected by a composite of p_value, AUC, and MRMR method")
                        else:
                            raise ValueError("FEATURE_SELECTION_METHOD is not correct. It should be 'mrmr', 'pvalue', 'auc', or 'composite'")

                        df = df_copy[exclude_columns + selected_features + [outcome_column]]

                        # =========================================
                        # Model building and evaluation
                        # =========================================
                        if MODEL_BUILDING_RADIOMICS:
                            eval_kwargs = {'test_size': TEST_SIZE_RADIOMICS,
                                           'random_state': 42} if EVALUATION_METHOD_RADIOMICS == 'train_test_split' else {'cv_folds': CV_FOLDS_RADIOMICS}

                            print("\n======================================================================")
                            print(f"Training and evaluating classification models for {num_features} feature(s) in sheet {sheet}")
                            print("======================================================================")
                            X = df.loc[:, ~df.columns.isin(exclude_columns + [outcome_column])]
                            y = df[outcome_column]

                            classification_results = evaluate_models(X, y, method=EVALUATION_METHOD_RADIOMICS, **eval_kwargs)

                            classification_results_file = os.path.join(result_dir, 'model_evaluation_results.xlsx')
                            save_classification_results(classification_results, classification_results_file, num_features, method=EVALUATION_METHOD_RADIOMICS)

                            # Record summary results
                            for classifier, result in classification_results.items():
                                result_entry = {
                                    'Sheet': sheet,
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


    # **************************************************************
    # **************************************************************
    # **************************************************************
    # **************************************************************


    if DEEP_LEANING:
        # Deep Feature Extraction
        if FEATURE_EXTRACTION_DEEP:
            cases = os.listdir(img_data_path)
            liver_features = []
            panc_features = []

            for case in cases:
                print(f"Processing {case}...")
                try:
                    case_folder = os.path.join(img_data_path, case)
                    nifti_files = os.listdir(case_folder)
                    ct_file = [f for f in nifti_files if f.startswith("DICOM") and f.endswith('nii')][0]
                    liver_file = [f for f in nifti_files if f.startswith("liver")][0]
                    panc_file = [f for f in nifti_files if f.startswith("pancreas")][0]
                    ct_path = os.path.join(case_folder, ct_file)
                    liver_path = os.path.join(case_folder, liver_file)
                    panc_path = os.path.join(case_folder, panc_file)

                    if not os.path.exists(liver_path):
                        print(f"file {liver_path} don't exist..!")
                        continue
                    elif not os.path.exists(panc_path):
                        print(f"file {panc_path} don't exist..!")
                        continue
                    elif not os.path.exists(ct_path):
                        print(f"file {ct_path} don't exist..!")
                        continue
                    else:
                        try:
                            l_features = extract_deep_features(ct_path, liver_path, dl_model_name)
                            liver_features_row = {'CaseNo': case}
                            liver_features_row.update(l_features)
                            liver_features.append(liver_features_row)

                            p_features = extract_deep_features(ct_path, panc_path, dl_model_name)
                            panc_features_row = {'CaseNo': case}
                            panc_features_row.update(p_features)
                            panc_features.append(panc_features_row)
                        except Exception as e:
                            print(f"Case {case} couldn't be processed because: ", e)

                except Exception as e1:
                    print(f"Case {case} couldn't be processed because: ", e1)

            df1 = pd.DataFrame(liver_features)
            df2 = pd.DataFrame(panc_features)
            writer = pd.ExcelWriter('deep_features1.xlsx', engine='xlsxwriter')
            df1.to_excel(writer, sheet_name='liver_dl_features', index=False)
            df2.to_excel(writer, sheet_name='pancreas_dl_features', index=False)
            writer.close()

        # DL feature selection and model building
        for excel_file_name in feature_file_names_dl:
            features_file = os.path.join(data_path, excel_file_name + ".xlsx")
            results_dir = os.path.join(result_path, excel_file_name)
            os.makedirs(results_dir, exist_ok=True)

            xls = pd.ExcelFile(features_file)
            summary_results = []
            best_result = None

            selected_sheets = []
            if SELECTED_SHEET_DL == "all":
                selected_sheets = xls.sheet_names
            else:
                selected_sheets.append(SELECTED_SHEET_DL)

            for sheet in selected_sheets:
                result_dir = os.path.join(results_dir, sheet)
                os.makedirs(result_dir, exist_ok=True)

                df = pd.read_excel(xls, sheet_name=sheet)

                # DL feature selection
                if FEATURE_CORRELATION_DL:
                    print("\n======================================================================")
                    print(f"Removing correlated features for sheet {sheet}")
                    print("======================================================================")
                    df = remove_collinear_features(df, CORR_THRESH_DL)

                if FEATURE_SELECTION_DL:
                    print("\n======================================================================")
                    print(f"Performing feature analysis for sheet {sheet}")
                    print("======================================================================")
                    p_values_df = calculate_p_values(df, outcome_column, categorical_columns, exclude_columns)
                    auc_values_df = calculate_auc_values_CV(df, outcome_column, categorical_columns,
                                                            exclude_columns)
                    mrmr_df = MRMR_feature_count(df, outcome_column, categorical_columns, exclude_columns,
                                                 max_num_features_dl, CV_FOLDS_DL)
                    composite_df = calculate_feature_scores(p_values_df, auc_values_df, mrmr_df, result_dir)

                    save_feature_analysis(p_values_df, auc_values_df, mrmr_df, composite_df, result_dir)

                    df_copy = df.copy()

                    for num_features in range(min_num_features_dl, max_num_features_dl + 1):
                        print("\n======================================================================")
                        print(f"Selecting {num_features} significant features for sheet {sheet}")
                        print("======================================================================")

                        selected_features = []
                        if FEATURE_SELECTION_METHOD_DL == 'mrmr':
                            selected_features = mrmr_df['Feature'][:num_features].tolist()
                            print(f"{num_features} features were selected by using MRMR method")
                        elif FEATURE_SELECTION_METHOD_DL == 'pvalue':
                            selected_features = p_values_df['Feature'][:num_features].tolist()
                            print(f"{num_features} features were selected by using pvalue method")
                        elif FEATURE_SELECTION_METHOD_DL == 'auc':
                            selected_features = auc_values_df['Feature'][:num_features].tolist()
                            print(f"{num_features} features were selected by using auc method")
                        elif FEATURE_SELECTION_METHOD_DL == 'composite':
                            selected_features = composite_df['Feature'][:num_features].tolist()
                            print(
                                f"{num_features} features were selected by a composite of p_value, AUC, and MRMR method")
                        else:
                            raise ValueError(
                                "FEATURE_SELECTION_METHOD is not correct. It should be 'mrmr', 'pvalue', 'auc', or 'composite'")

                        df = df_copy[exclude_columns + selected_features + [outcome_column]]

                        # =========================================
                        # Model building and evaluation
                        # =========================================
                        if MODEL_BUILDING_DL:
                            eval_kwargs = {'test_size': TEST_SIZE_DL,
                                           'random_state': 42} if EVALUATION_METHOD_DL == 'train_test_split' else {
                                'cv_folds': CV_FOLDS_DL}

                            print("\n======================================================================")
                            print(
                                f"Training and evaluating classification models for {num_features} feature(s) in sheet {sheet}")
                            print("======================================================================")
                            X = df.loc[:, ~df.columns.isin(exclude_columns + [outcome_column])]
                            y = df[outcome_column]

                            classification_results = evaluate_models(X, y, method=EVALUATION_METHOD_DL,
                                                                     **eval_kwargs)

                            classification_results_file = os.path.join(result_dir, 'model_evaluation_results.xlsx')
                            save_classification_results(classification_results, classification_results_file,
                                                        num_features, method=EVALUATION_METHOD_DL)

                            # Record summary results
                            for classifier, result in classification_results.items():
                                result_entry = {
                                    'Sheet': sheet,
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
    faulthandler.enable()
    main()

