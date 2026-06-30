# VENDORED, verbatim, from lamf_analysis.code_ocean.json_utils
# (github.com/AllenNeuralDynamics/lamf-analysis). Vendored rather than installed because
# lamf-analysis pulls a heavy/conflict-prone dep tree (roicat->torch, dask/distributed pins,
# boto3/codeocean/s3fs pins, and an UNPINNED aind-data-schema that fights the ==1.2.0 needed
# here). This module is self-contained: it imports only aind_data_schema(+models) + stdlib.
# Keep in sync with upstream. Requires aind-data-schema==1.2.0 (pinned in environment/).
# aind-data-schema==1.2.0
import aind_data_schema
assert aind_data_schema.__version__ == '1.2.0'

# pylint: disable=pointless-string-statement
# pylint: disable=wrong-import-position
# pylint: disable=wrong-import-order

''' Example (in run_capsule.py)

from lamf_analysis.code_ocean import json_utils
import aind_data_schema
assert(aind_data_schema.__version__ == '1.2.0')

SUFFIX = 'CTL-czstack-to-fov'
PROCESS_LEVEL = 'subject' # 'subject' or 'session'

INPUT_PROCESSING_DICT = {"name": SUFFIX,
                         "software_version": "0.1.0",
                         "code_url": "https://codeocean.allenneuraldynamics.org/capsule/2051964/tree",
                         "notes": 'Cell Type and Learning cortical z-stack to FOV mapping'}

if __name__ == "__main__":
    start_date_time = datetime.datetime.now()

    source_asset_name = "..."
    capture_name = "..."
    run_parameters = {"...": ...,
                     }

    json_utils.process_json_files(source_asset_name,
                                    capture_name,
                                    start_date_time,
                                    run_parameters,
                                    INPUT_PROCESSING_DICT,
                                    SUFFIX,
                                    PROCESS_LEVEL
                                    )
'''

from pathlib import Path
import datetime
import shutil
import json

from aind_data_schema.core.data_description import (
    DataDescription,
    DerivedDataDescription,
    DataLevel,
    Organization,
    Modality,
    Platform,
    Funding,
)
from aind_data_schema.core.processing import (
    DataProcess,
    Processing,
    PipelineProcess)
from aind_data_schema_models.pid_names import PIDName

DATA_PATH = Path('/root/capsule/data/')
RESULTS_PATH = Path('/root/capsule/results/')

def copy_core_json(source_asset_name):
    # glob (one level), NOT rglob: the core metadata jsons live at the asset root
    # (DATA_PATH/<asset>/<x>.json). rglob would recurse into the segmentation zarr (10k+
    # files) and stall the run. VENDORED-LOCAL FIX vs upstream lamf_analysis.json_utils.
    session_json_path = next(DATA_PATH.glob(f'*{source_asset_name}*/*session.json'), None)
    procedures_json_path = next(DATA_PATH.glob(f'{source_asset_name}*/*procedures.json'), None)
    subject_json_path = next(DATA_PATH.glob(f'*{source_asset_name}*/*subject.json'), None)
    rig_json_path = next(DATA_PATH.glob(f'*{source_asset_name}*/*rig.json'), None)

    if not session_json_path:
        print('No session json found')
    else:
        shutil.copy(session_json_path.as_posix(), (RESULTS_PATH / 'session.json').as_posix())

    if not subject_json_path:
        print('No subject json found')
    else:
        shutil.copy(subject_json_path.as_posix(), (RESULTS_PATH / 'subject.json').as_posix())

    if not procedures_json_path:
        print('No procedures json found')
    else:
        shutil.copy(procedures_json_path.as_posix(), (RESULTS_PATH / 'procedures.json').as_posix())

    if not rig_json_path:
        print('No rig json found, trying instrument json')
        instrument_json_path = next(DATA_PATH.glob(f'*{source_asset_name}*/*instrument.json'), None)
        if not instrument_json_path:
            print('No instrument json found')
        else:
            shutil.copy(instrument_json_path.as_posix(), (RESULTS_PATH / 'instrument.json').as_posix())
    else:
        shutil.copy(rig_json_path.as_posix(), (RESULTS_PATH / 'rig.json').as_posix())


def process_json_files(source_asset_name: str, # data asset name of the source
                       capture_name: str, # must be '*_*_{0-9:4}-{0-9:2}-{0-9:2}_{0-9:2}-{0-9:2}-{0-9:2}*' format,
                       start_date_time: datetime.datetime,
                       run_parameters: dict,
                       input_processing_dict: dict, # name, version, code_url, notes
                       process_name: str,
                       process_level: str, # 'subject', 'session', etc.
                       processor_full_name: str="Jinho Kim",
                       ):
    subject_id = source_asset_name.split('_')[1]

    if process_level != 'subject':
        copy_core_json(source_asset_name)

    data_description_json_path = next(DATA_PATH.glob(f'*{source_asset_name}*/data_description.json'), None)
    if data_description_json_path is not None:
        with data_description_json_path.open('r') as f:
            processed_data_description_json = json.load(f)
    else:
        # Some HCR processed assets ship WITHOUT a data_description.json. Fall back to the shipped
        # sample HCR data_description (correct SPIM/HCR institution / platform / modality / funding /
        # investigators) rather than upstream's base_data_description_json(), which hardcodes
        # POPHYS / MULTIPLANE_OPHYS and would mislabel the asset. Per-asset fields are fixed up:
        # subject_id here; name + creation_time in get_data_description_dict; lineage/input comes from
        # capture_name in DerivedDataDescription.from_data_description (so the sample's subject_id /
        # name / input_data_name are NOT inherited).
        sample = Path(__file__).parent / 'sample_HCR_data_description.json'
        if not sample.exists():
            raise FileNotFoundError(
                f"source data_description.json not found for '{source_asset_name}' under {DATA_PATH}, "
                f"and no fallback sample at {sample}. Attach the processed-HCR source asset, or ship "
                f"sample_HCR_data_description.json next to json_utils.py.")
        print(f"No data_description.json in source asset; using fallback sample "
              f"'{sample.name}' (subject_id -> {subject_id})")
        with sample.open('r') as f:
            processed_data_description_json = json.load(f)
        processed_data_description_json['subject_id'] = subject_id  # the REAL subject, not the sample's
    
    end_date_time = datetime.datetime.now()
    data_description_dict = get_data_description_dict(capture_name, source_asset_name, processed_data_description_json)
    data_description = DataDescription(**data_description_dict)

    derived_data_description = DerivedDataDescription.from_data_description(
        data_description=data_description, process_name=process_name
    )
    with (RESULTS_PATH / "data_description.json").open("w") as f:
        f.write(derived_data_description.model_dump_json(indent=3))

    processing_dict = get_processing_dict(start_date_time, end_date_time, run_parameters, input_processing_dict)
    processing_model = DataProcess(**processing_dict)
    processing_pipeline = PipelineProcess(data_processes = [processing_model], processor_full_name=processor_full_name)
    processing = Processing(processing_pipeline=processing_pipeline)
    processing.write_standard_file(RESULTS_PATH)


def base_data_description_json(subject_id: str) -> dict:
    base_data_description = {
        "institution": Organization.AIND,
        "investigators": [PIDName(name="Unknown")],
        "funding_source": [Funding(funder=Organization.AI)],
        "modality": [Modality.POPHYS],
        "platform": Platform.MULTIPLANE_OPHYS,
        "subject_id": subject_id,
    }
    return base_data_description


def get_data_description_dict(capture_name, source_asset_name, processed_data_description_json) -> dict:

    data_description_dict = {}
    copy_keys = ['institution', 'investigators', 'funding_source', 'modality', 'platform', 'subject_id']
    for key in copy_keys:
        if key in processed_data_description_json:
            data_description_dict[key] = processed_data_description_json[key]
        else:
            print(f"Warning: {key} not found in processed_data_description_json")
            data_description_dict[key] = None
    data_description_dict["creation_time"] = datetime.datetime.now()
    data_description_dict["name"] = capture_name
    data_description_dict["data_level"] = DataLevel.DERIVED
    # data_description_dict["input_data_name"] = source_asset_name
    
    return data_description_dict


def get_processing_dict(start_date_time: datetime.datetime, end_date_time: datetime.datetime, run_parameters: dict,
                        input_processing_dict: dict) -> dict:
    data_processing_dict = {}
    data_processing_dict["name"] = input_processing_dict['name']
    data_processing_dict["software_version"] = input_processing_dict['software_version']
    data_processing_dict["start_date_time"] = str(start_date_time)
    data_processing_dict["end_date_time"] = str(end_date_time)
    data_processing_dict["input_location"] = DATA_PATH.as_posix()
    data_processing_dict["output_location"] = RESULTS_PATH.as_posix()
    data_processing_dict["code_url"] = input_processing_dict['code_url']
    data_processing_dict["parameters"] = run_parameters
    data_processing_dict["notes"] = input_processing_dict['notes']
    data_processing_dict["outputs"] = {}

    return data_processing_dict