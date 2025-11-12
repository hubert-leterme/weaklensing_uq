This is a brief description of the HST/COSMOS
weak lensing reduced shear catalog from Schrabback et al. (2010, A&A 516, A63).
For more details please see the original paper. If you use the
catalog in a publication please acknowledge the use by citing this paper.
The matched photo-z information in the bright catalog
is provided for convenience and to make it easier to reproduce the
results from Schrabback et al. (2010). These photo-z measurements
originate from the catalog released by Ilbert et al. (2009, ApJ 690, Issue 2, pp. 1236-1249). In case you make use of this information please make sure to cite
this paper as well.

An example acknowledgement could be: "This article makes use of the
weak lensing shear catalog created by Schrabback et al. (2010, A&A 516, A63),
which was based on observations conducted by the NASA/ESA Hubble Space
Telescope targeting the COSMOS field (Scoville et al. 2007, ApJS, 172, 1, pp. 38-45.), as well as photometric redshift measurements from
Ilbert et al. (2009, ApJ 690, Issue 2, pp. 1236-1249)."

To use the catalog as done in Schrabback et al. (2010) there are
two relevant catalog sub-sets: 1. the galaxies with COSMOS-30 photoz (imag<25, "bright")
and 2. those without individual photoz ("faint", imag>25 or in
a masked region of the COSMOS30 catalog). 
There is no object in both catalogs, hence you may consider to combine 
them.

You could also consider to employ more recent updated photometric redshift
information, e.g. from  Laigle et al. (2016, ApJS, 224, 2, A24).

The "faint" catalog also includes quite noisy sources (down to MAG_AUTO<26.7).
For some applications it might be useful to apply a more conservative selection.

1. Bright:
----------
cosmos_bright_cat_min.asc

Most important columns:

#   1 Ra                  Sky coordinates [J2000]
#   2 Dec                 Sky coordinates [J2000]
#   3 Xs                  Mosaic X-Y coordinates (North up, East left)
#   4 Ys                  with pixel scale 0.186''. Can directly be used in combination
#                         with the shear estimate keys  e1iso_snCal_rot4  e2iso_snCal_rot4            
#   5 MAG_AUTO            Kron-like elliptical aperture magnitude from SEXtractor        [AB mag]   
#   6 e1iso_rot4_gr_snCal BEST g1 shear estimate for North=up,  all fixes included   
#   7 e2iso_rot4_gr_snCal BEST g2 shear estimate for North=up,  all fixes included
#  12 nhweight_int        shear-measurement weight
#  20 zphot               Photoz estimate from Ilbert et al. (2009)
#  21 z_problem           Potentially high redshift (zphot<0.6, MAG_AUTO>24)                     
#                         0=fine, 1=potentially problematic (i>24), 2=likely problematic (i>24, zphot_sec>0)                    

For a proper redshift calibration it is recommended to exclude potentially problematic objects by selecting galaxies with z_problem=0 or z_problem<=1 (in the latter case you only exclude those galaxies with a sig. secondary redshift peak, which is probably sufficient).

 
2. Faint
--------

Catalog:

cosmos_faint_cat.asc

Keys as above but no redshifts!

Redshift distribution for this "faint catalog":

Uniform shear weights:
cosmos_zdist_faint_w0.asc

Shear weights nhweight_int applied:
cosmos_zdist_faint_w1.asc

They are normalised such that 
Sum(N(z))=100*(Number galaxies in faint catalog)

-----------

Regarding E/B-modes:

The plots in Schrabback et al. (2010) are for the combined catalog.
It seems that B-modes are slightly more significant if only the "bright"
galaxies are used, but these are heavily masked from the ground-based
catalog, potentially causing issues.

Please contact Tim Schrabback (schrabba@astro.uni-bonn.de) in case you
have any questions, or if you would like to inform us about your work,
which we would appreciate very much.


