import argparse, textwrap
import os
from pathlib import Path
import torch
import sys
import types
import multiprocessing as mp

# This is just to silence nnUNet warnings. These variables should have no purpose/effect.
# There are sadly no other workarounds at the moment, see:
# https://github.com/MIC-DKFZ/nnUNet/blob/227d68e77f00ec8792405bc1c62a88ddca714697/nnunetv2/paths.py#L21
os.environ['nnUNet_raw'] = "./nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "./nnUNet_preprocessed"
os.environ['nnUNet_results'] = "./nnUNet_results"

from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def main():
    # Description and arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            This script runs nnUNetV2 inference. Based on https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/predict_from_raw_data.py.
        '''.split()),
        epilog=textwrap.dedent('''
            Example:
            predict_nnunet -i in_folder -o out_folder -d 101 -c 3d_fullres -p nnUNetPlans_small -tr nnUNetTrainer_DASegOrd0_NoMirroring -f 0
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--images-dir', '-i', type=Path, required=True,
        help='The folder where input NIfTI images files are located (required).'
    )
    parser.add_argument(
        '--output-dir', '-o', type=Path, required=True,
        help='The folder where nnUNet predictions will be stored (required).'
    )
    parser.add_argument(
        '--dataset', '-d', type=str, required=True,
        help='nnUNet dataset number, example 567 for Dataset567_... folder under nnunet_results (required).'
    )
    parser.add_argument(
        '--configuration', '-c', type=str, required=True,
        help='nnUNet configuration'
    )
    parser.add_argument(
        '--plans', '-p', type=str, default='nnUNetPlans',
        help='nnUNet plans, default is "nnUNetPlans".'
    )
    parser.add_argument(
        '--trainer', '-tr', type=str, default='nnUNetTrainer',
        help='nnUNet trainer, default is "nnUNetTrainer".'
    )
    parser.add_argument(
        '--folds', '-f', nargs='+', type=str, default=(0, 1, 2, 3, 4),
        help='nnUNet folds, default is "(0, 1, 2, 3, 4)".'
    )
    parser.add_argument(
        '-step-size', type=float, default=0.5,
        help='Step size for sliding window prediction, default is "0.5".'
    )
    parser.add_argument(
        '--disable-tta', action='store_true', default=False,
        help='Set this flag to disable test time data augmentation, default is false.'
    )
    parser.add_argument(
        '--verbose', action='store_true', 
        help="Display extra information, defaults to false (display)."
    )
    parser.add_argument(
        '--save-probabilities', action='store_true',
        help='Set this to export predicted class "probabilities", default is false'
    )
    parser.add_argument(
        '--continue-prediction', action='store_true',
        help='Continue an aborted previous prediction (will not overwrite existing files), default is false'
    )
    parser.add_argument(
        '-chk', type=str, default='checkpoint_final.pth',
        help='Name of the checkpoint you want to use, default is "checkpoint_final.pth".'
    )
    parser.add_argument(
        '-npp', type=int, default=3,
        help='Number of processes used for preprocessing, default is "3".'
    )
    parser.add_argument(
        '-nps', type=int, default=3,
        help='Number of processes used for segmentation export, default is "3".'
    )
    parser.add_argument(
        '-prev-stage-predictions', type=str, default=None,
        help='Folder containing the predictions of the previous stage. Required for cascaded models, default is None'
    )
    parser.add_argument(
        '-num-parts', type=int, default=1,
        help='Number of separate nnUNetv2_predict call that you will be making, default is "1".'
    )
    parser.add_argument(
        '-part-id', type=int, required=False, default=0,
        help='If multiple nnUNetv2_predict exist, default is "0".'
    )
    parser.add_argument(
        '-device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Use this to set the device the inference should run with, default is "cuda".'
    )
    parser.add_argument(
        '--disable-progress-bar', action='store_true',
        help='Set this flag to disable progress bar, default is false.'
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get the command-line argument values
    images_dir = args.images_dir
    output_dir = args.output_dir
    dataset = args.dataset
    configuration = args.configuration
    plans = args.plans
    trainer = args.trainer
    folds = args.folds
    step_size = args.step_size
    disable_tta = args.disable_tta
    save_probabilities = args.save_probabilities
    continue_prediction = args.continue_prediction
    checkpoint = args.chk
    npp = args.npp
    nps = args.nps
    prev_stage_predictions = args.prev_stage_predictions
    num_parts = args.num_parts
    part_id = args.part_id
    device = args.device
    verbose = args.verbose
    disable_progress_bar = args.disable_progress_bar

    # Get model folder
    model_folder = get_output_folder(dataset, trainer, plans, configuration)

    # Print the argument values if not quiet
    if verbose:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            images_dir = "{images_dir}"
            output_dir = "{output_dir}"
            dataset = "{dataset}"
            configuration = "{configuration}"
            plans = "{plans}"
            trainer = "{trainer}"
            folds = "{folds}"
            step_size = "{step_size}"
            disable_tta = "{disable_tta}"
            save_probabilities = "{save_probabilities}"
            continue_prediction = "{continue_prediction}"
            checkpoint = "{checkpoint}"
            npp = "{npp}"
            nps = "{nps}"
            prev_stage_predictions = "{prev_stage_predictions}"
            num_parts = "{num_parts}"
            part_id = "{part_id}"
            device = "{device}"
            verbose = "{verbose}"
            disable_progress_bar = "{disable_progress_bar}"
        '''))
    
    # Load device
    if isinstance(device, str):
        assert device in ['cpu', 'cuda', 'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {device}.'
        if device == 'cpu':
            # let's allow torch to use hella threads
            import multiprocessing
            torch.set_num_threads(multiprocessing.cpu_count())
            device = torch.device('cpu')
        elif device == 'cuda':
            # multithreading in torch doesn't help nnU-Net if run on GPU
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            device = torch.device('cuda')
        else:
            device = torch.device('mps')

    predict_nnunet(
        model_folder = model_folder,
        images_dir = images_dir,
        output_dir = output_dir,
        folds = folds,
        step_size = step_size,
        disable_tta = disable_tta,
        save_probabilities = save_probabilities,
        continue_prediction = continue_prediction,
        checkpoint = checkpoint,
        npp = npp,
        nps = nps,
        prev_stage_predictions = prev_stage_predictions,
        num_parts = num_parts,
        part_id = part_id,
        device = device,
        verbose = verbose,
        disable_progress_bar = disable_progress_bar
    )


def predict_nnunet(
        model_folder,
        images_dir,
        output_dir,
        device, # torch device
        folds = (0, 1, 2, 3, 4),
        step_size = 0.5,
        disable_tta = False,
        save_probabilities = False,
        continue_prediction = False,
        checkpoint = 'checkpoint_final.pth',
        npp = 3,
        nps = 3,
        prev_stage_predictions = None,
        num_parts = 1,
        part_id = 0,
        verbose = False,
        disable_progress_bar = False
):
    # nnUNet uses multiprocessing background workers for preprocessing/export.
    # On Linux, default 'fork' can be fragile after CUDA initialization and may
    # lead to silent worker deaths/hangs. Force 'spawn' where possible.
    try:
        mp.set_start_method('spawn', force=False)
    except RuntimeError:
        pass

    # SimpleITK/ITK can oversubscribe threads inside each worker, amplifying RAM
    # usage and causing workers to be killed. Keep it single-threaded by default.
    os.environ.setdefault('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

    # Be conservative by default: nnUNet spawns processes and each can be heavy.
    try:
        npp = max(1, int(npp))
        nps = max(1, int(nps))
    except Exception:
        npp, nps = 1, 1
    npp = min(npp, 2)
    nps = min(nps, 2)

    # Check variables
    folds = [i if i == 'all' else int(i) for i in folds]
    assert part_id < num_parts

    # Convert to Path objects if strings
    from pathlib import Path
    model_folder = Path(model_folder) if isinstance(model_folder, str) else model_folder
    images_dir = Path(images_dir) if isinstance(images_dir, str) else images_dir
    output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir

    # Create output folder if does not exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --------------------------------------------------------------------------------------------------------------
    # nnUNetv2 trainer discovery workaround
    #
    # nnUNetv2 uses recursive module imports under nnunetv2/training/nnUNetTrainer to find trainer classes.
    # If ANY trainer module in that folder raises ImportError at import time, inference can fail even if the
    # requested trainer is unrelated (this happens in some environments with leftover LDH Step6 trainers).
    #
    # To make inference robust, we pre-stub known-bad modules so importlib returns a harmless module instead
    # of executing the faulty file.
    # --------------------------------------------------------------------------------------------------------------
    _bad_trainer_modules = [
        "nnunetv2.training.nnUNetTrainer.nnUNetTrainer_LDH_Step6",
    ]
    for _m in _bad_trainer_modules:
        if _m not in sys.modules:
            sys.modules[_m] = types.ModuleType(_m)
    
    # Start nnUNet inference
    predictor = nnUNetPredictor(tile_step_size=step_size,
                                use_gaussian=True,
                                use_mirroring=not disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=verbose,
                                verbose_preprocessing=verbose,
                                allow_tqdm=not disable_progress_bar)
    
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        folds,
        checkpoint_name=checkpoint
    )
    predictor.predict_from_files(str(images_dir), str(output_dir), save_probabilities=save_probabilities,
                                 overwrite=not continue_prediction,
                                 num_processes_preprocessing=npp,
                                 num_processes_segmentation_export=nps,
                                 folder_with_segs_from_prev_stage=prev_stage_predictions,
                                 num_parts=num_parts,
                                 part_id=part_id)

if __name__=='__main__':
    main()