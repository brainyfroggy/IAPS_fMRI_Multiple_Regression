import gl


MNI152 = "C:/Program Files/MRIcroGL/Resources/standard/mni152.nii.gz"
UNIVARIATE = (
    "N:/Experimental_Data/yujunchen/projects/IAPS_fMRI_multiple_reg/outputs/"
    "valence_arousal_both_positive_significant_min_t.nii.gz"
)
MULTIVARIATE = (
    "N:/Experimental_Data/yujunchen/projects/IAPS_Searchlight/outputs/tBrainmap/"
    "final_results/tmap_beh60_p05_v2.nii.gz"
)
OUTPUT = (
    "N:/Experimental_Data/yujunchen/projects/IAPS_fMRI_multiple_reg/figures/"
    "mricrogl_mni152_univariate_multivariate_raw.png"
)


gl.resetdefaults()
gl.scriptformvisible(0)
gl.toolformvisible(0)
gl.windowposition(20, 20, 1800, 980)
gl.backcolor(0, 0, 0)
gl.loadimage(MNI152)
gl.minmax(0, 10, 90)
gl.smooth(1)

# Smooth trilinear reslicing removes the coarse 3-mm voxel stair-step display.
gl.overlayloadsmooth(1)
gl.overlayload(UNIVARIATE)
gl.minmax(1, 2.81, 6.48)
gl.colorname(1, "5winter")
gl.colorfromzero(1, 0)
gl.opacity(1, 100)

gl.overlayloadsmooth(1)
gl.overlayload(MULTIVARIATE)
gl.minmax(2, 3.52, 7.56)
gl.colorname(2, "4hot")
gl.colorfromzero(2, 0)
gl.opacity(2, 90)

gl.overlayadditiveblending(0)
gl.colorbarposition(0)
gl.bmpzoom(2)
gl.mosaic("A L+ H -0.05 -11 -5 1 7; 13 19 31 64")
gl.wait(1000)
gl.savebmp(OUTPUT)
gl.quit()
