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
OUTPUT_DIR = (
    "N:/Experimental_Data/yujunchen/projects/IAPS_fMRI_multiple_reg/figures/"
    "mricrogl_axial_slices/"
)
Z_COORDINATES = (-8, -4, 0, 4, 8, 12, 16, 66)


gl.resetdefaults()
gl.scriptformvisible(0)
gl.toolformvisible(0)
gl.windowposition(20, 20, 760, 760)
gl.backcolor(255, 255, 255)
gl.loadimage(MNI152)
gl.minmax(0, 10, 90)
gl.smooth(1)

gl.overlayloadsmooth(1)
gl.overlayload(MULTIVARIATE)
gl.minmax(1, 2.0, 7.0)
gl.colorname(1, "4hot")
gl.colorfromzero(1, 0)
gl.opacity(1, 90)

gl.overlayloadsmooth(1)
gl.overlayload(UNIVARIATE)
gl.minmax(2, 2.0, 7.0)
gl.colorname(2, "5winter")
gl.colorfromzero(2, 0)
gl.opacity(2, 100)

gl.overlayadditiveblending(0)
gl.colorbarposition(0)
gl.bmpzoom(2)

for z_coordinate in Z_COORDINATES:
    gl.mosaic("A " + str(z_coordinate))
    gl.wait(250)
    label = "neg" + str(abs(z_coordinate)) if z_coordinate < 0 else "pos" + str(z_coordinate)
    gl.savebmp(OUTPUT_DIR + "z_" + label + ".png")

gl.quit()
