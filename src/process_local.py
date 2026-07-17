"""Local entrypoint for running the PANORAMA inference workflow.

The Docker/challenge entrypoint is ``process.py``. This module keeps a local
workflow available without duplicating inference logic or baking in lab paths.
"""

from process import PDACDetectionContainer, parse_args


def main():
    args = parse_args()
    PDACDetectionContainer(
        nnunet_base=args.nnunet_base,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        image_ext=args.image_ext,
    ).process()


if __name__ == "__main__":
    main()
