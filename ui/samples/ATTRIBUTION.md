# Sample image attribution

The four sample X-rays in this folder come from the held-out **test** split of:

> Kermany, D., Zhang, K., Goldbaum, M. (2018). *Labeled Optical Coherence
> Tomography (OCT) and Chest X-Ray Images for Classification.* Mendeley Data,
> v2. https://doi.org/10.17632/rscbjbr9sj.2

License: **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/).
They were redistributed via the Kaggle mirror
`paultimothymooney/chest-xray-pneumonia`.

Modifications: converted to grayscale, downscaled to a maximum side of 1024 px,
and re-encoded as JPEG. They were selected at random (seed 2026, 2 per class),
not chosen for model performance. `samples.json` maps each file to its original
filename.
