#### Project Questions/Remarks
**1.** We apply White balance algorithms only in linear RGB because they rely on physical assumptions about scene illumination. These assumptions are only valid when working with actual radiance values, which linear RGB represents.
For sRGB images, pixel values are non-linear. Therefore, they would require preprocessing by conversion from sRGB to linear RGB before applying white balance corrections. After correction, we postprocess by converting back from linear RGB to sRGB.

-----

**2.** 
