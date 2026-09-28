import os
import shutil
import tempfile
import zipfile
import pandas as pd
import pydicom
import requests
import SimpleITK as sitk

# ----------------- CONFIGURATION -----------------
ORTHANC_URL = "https://pacs-ayurveda.radpretation.ai"
ORTHANC_USER = None
ORTHANC_PASSWORD = None

EXCEL_FILE = r"dicom_to_nifti.py"
SHEET_NAME = 0
STUDY_ID_COL = "studyID"
OUTPUT_DIR = "Nifti_folder"
# --------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
auth = (ORTHANC_USER, ORTHANC_PASSWORD) if ORTHANC_USER and ORTHANC_PASSWORD else None


def resolve_orthanc_study_id(study_id: str) -> str:
    """Checks whether study_id is an internal Orthanc ID or a DICOM StudyInstanceUID."""
    # 1. Test direct Orthanc internal ID
    check_resp = requests.get(f"{ORTHANC_URL}/studies/{study_id}", auth=auth)
    if check_resp.status_code == 200:
        return study_id

    # 2. Query via StudyInstanceUID lookup
    payload = {
        "Level": "Study",
        "Query": {"StudyInstanceUID": study_id}
    }
    find_resp = requests.post(f"{ORTHANC_URL}/tools/find", json=payload, auth=auth)
    if find_resp.status_code == 200:
        results = find_resp.json()
        if results:
            return results[0]

    return None


def download_study_dicom(orthanc_id: str, extract_to: str):
    """Downloads the ZIP archive of the study and unpacks it."""
    archive_url = f"{ORTHANC_URL}/studies/{orthanc_id}/archive"
    with requests.get(archive_url, auth=auth, stream=True) as resp:
        resp.raise_for_status()
        zip_path = os.path.join(extract_to, "study.zip")
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    os.remove(zip_path)


def convert_study_to_nifti(dicom_root_dir: str, output_path: str):
    """
    Groups DICOM slices by SeriesInstanceUID and converts the primary
    series to NIfTI format using SimpleITK.
    """
    series_map = {}

    # 1. Walk through all extracted files and group valid DICOM image files
    for root, _, files in os.walk(dicom_root_dir):
        for f in files:
            file_path = os.path.join(root, f)
            try:
                # Read header only (fast)
                dcm = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
                if not hasattr(dcm, "SeriesInstanceUID"):
                    continue

                # Exclude non-image modalities (Structured Reports, Presentation States, etc.)
                modality = getattr(dcm, "Modality", "")
                if modality in ["SR", "PR", "KO", "DOC"]:
                    continue

                suid = str(dcm.SeriesInstanceUID)
                series_map.setdefault(suid, []).append(file_path)
            except Exception:
                continue

    if not series_map:
        raise RuntimeError("No readable DICOM image files found in the downloaded archive.")

    # 2. Pick the series with the largest number of slices (the main volume)
    best_series_uid = max(series_map, key=lambda k: len(series_map[k]))
    selected_files = series_map[best_series_uid]

    reader = sitk.ImageSeriesReader()

    # 3. If multi-slice volume (3D), sort slices spatially; otherwise load single slice (2D)
    if len(selected_files) > 1:
        # Find directory where files reside to let SimpleITK sort correctly
        series_dir = os.path.dirname(selected_files[0])
        sorted_filenames = reader.GetGDCMSeriesFileNames(series_dir, best_series_uid)
        
        # If files were spread across subfolders, fall back to the collected file list
        if not sorted_filenames:
            sorted_filenames = selected_files

        reader.SetFileNames(sorted_filenames)
        image = reader.Execute()
    else:
        image = sitk.ReadImage(selected_files[0])

    # 4. Save to final .nii.gz file
    sitk.WriteImage(image, output_path)


def process_all_studies():
    df = pd.read_excel(
        EXCEL_FILE,
        sheet_name=SHEET_NAME,
        dtype={STUDY_ID_COL: str},
        engine="openpyxl"
    )

    study_ids = df[STUDY_ID_COL].dropna().astype(str).tolist()

    for idx, raw_id in enumerate(study_ids, 1):
        clean_id = raw_id.strip()
        final_nifti_path = os.path.join(OUTPUT_DIR, f"{clean_id}.nii.gz")

        print(f"[{idx}/{len(study_ids)}] Processing: {clean_id}...")

        if os.path.exists(final_nifti_path):
            print(f"  -> File {final_nifti_path} already exists. Skipping.")
            continue

        orthanc_id = resolve_orthanc_study_id(clean_id)
        if not orthanc_id:
            print(f"  -> Study '{clean_id}' not found on Orthanc. Skipping.")
            continue

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                download_study_dicom(orthanc_id, temp_dir)
                convert_study_to_nifti(temp_dir, final_nifti_path)
                print(f"  -> Successfully saved: {final_nifti_path}")
            except Exception as e:
                print(f"  -> Failed to convert {clean_id}: {e}")


if __name__ == "__main__":
    process_all_studies()