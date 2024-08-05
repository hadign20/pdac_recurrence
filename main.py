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
excel_file_name = "JayaTexturePancreasMedian" # JayaTexturePancreasMedian, JayaTextureLiverMedian
SELECTED_SHEET = "all" #"2_1"
outcome_column = "EarlyRecurrence"
exclude_columns = ["CaseNo"]
categorical_columns = []


features_file = os.path.join(data_path, excel_file_name + ".xlsx")
results_dir = os.path.join(result_path, excel_file_name)
os.makedirs(results_dir, exist_ok=True)

#=========================================
# set parameters
#=========================================
FEATURE_EXTRACTION_RADIOMICS = False
FEATURE_EXTRACTION_DEEP = True

data_path = r'D:\projects\pdac_reproducibility\PDACreproducibility'
result_path = r'D:\projects\pdac_reproducibility\pdac_reproducibility\results'
params_path = r'D:\projects\pdac_reproducibility\pdac_reproducibility\src\feature_extraction\CT.yaml'

image_dir = r'D:\projects\PDAC Recurrence\data\scans'
dl_excel_path = os.path.join(result_path, 'deep_features.xlsx')
dl_model_name = 'resnet50'  # ('vgg16', 'densenet121')


FEATURE_CORRELATION = 0
CORR_THRESH = 0.8

FEATURE_SELECTION = 0
FEATURE_SELECTION_METHOD = 'composite' # 'mrmr', 'pvalue', 'auc', 'composite'
min_num_features = 1
max_num_features = 20

MODEL_BUILDING = 0
EVALUATION_METHOD = 'train_test_split' # 'train_test_split' or 'cross_validation'
TEST_SIZE = 0.3
CV_FOLDS = 10
HYPERPARAMETER_TUNING = False




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
    # =========================================================
    # Deep Feature Extraction
    # =========================================================
    if FEATURE_EXTRACTION_DEEP:
        cases = os.listdir(image_dir)
        liver_features = []
        panc_features = []

        for case in cases:
            print(f"Processing {case}...")
            try:
                case_folder = os.path.join(image_dir, case)
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


if __name__ == '__main__':
    faulthandler.enable()
    main()

