# Voxelwise valence/arousal regression

## Model

The analysis uses the available `allsub_avg.npy` beta matrix, which has shape `20 x 60 x 153594`: 20 subjects, 60 stimulus-averaged beta patterns, and 153594 voxels. The model fit at each subject/voxel is:

`fMRI beta = intercept + z(valence) + z(arousal)`

The output maps are group summaries across the 20 subject-level slopes.

## Why 60 averaged stimuli instead of 300 trials?

The current reusable data in `IAPS_Searchlight` are already averaged to 60 stimuli. This is also the better simple model for these predictors: valence and arousal are properties of the stimulus, so repeating the same rating 5 times and treating the 300 rows as independent would inflate the apparent degrees of freedom unless the model explicitly handled stimulus/repetition/subject dependence. For a simple voxelwise regression, the 60 averaged stimulus betas are the cleaner choice.

If you want to use all 300 single-trial betas later, the better version would be a mixed-effects or two-stage model that accounts for repeated presentations rather than a plain OLS on 300 rows.

## Design diagnostics

- Valence/arousal predictor correlation: `-0.271`
- VIF for each predictor in the two-predictor model: `1.079`

## Visualization recommendation

Use unthresholded group mean beta maps to show effect direction and scale, and group t maps to show reliability across subjects. For a paper figure, the most interpretable static view is usually a small set of anatomically meaningful slices from the t maps, with matching unthresholded beta maps in the supplement. The projection PNGs here are useful as a fast whole-brain overview, but slice maps are better for localization.

For final reporting, the strongest next visualization would be thresholded group t maps on an anatomical template with a diverging colormap centered at zero, plus ROI summaries or scatterplots from peak/ROI voxels to show how beta changes with valence and arousal.
