#### Project Questions/Remarks
**1.** We apply White balance algorithms only in linear RGB because they rely on physical assumptions about scene illumination. These assumptions are only valid when working with actual radiance values, which linear RGB represents.
For sRGB images, pixel values are non-linear. Therefore, they would require preprocessing by conversion from sRGB to linear RGB before applying white balance corrections. After correction, we postprocess by converting back from linear RGB to sRGB.

-----
**2.**
img1: main light source: moon, single light and indirect light. yes they are cues, the moon reflect light from the sun.
img2: main light source: light polution from city in the background, multiple lights stars, tower and city. yes they are cues, light on the horizon.  
img3: main light source: sun, waterfall reflected light. yes they are cues, sunbeams from the cource. 
img4: main light sources: stars, multiple scources, no they are to many scources.xxw
img5: main light source: flash in the clouds, multiple flash and stars. yes they are cues, areas in the clouds they reflect the flash. 

-----

**3.** 
**Eye movements** fall into two functional categories: gaze-stabilization and gaze-shifting movements. Gaze-stabilization movements, which prevent images from slipping across the retina during head or body motion, rely on the vestibulo-ocular and optokinetic systems. Vestibulo-ocular movements (VOR) use vestibular input to produce rapid compensatory eye movements opposite the direction of head movement; they work best for fast, brief motions but diminish after prolonged rotation unless visual cues are present. Optokinetic movements, by contrast, stabilize vision in response to sustained motion of the visual field and produce optokinetic nystagmus—a combination of slow pursuit of moving scenery and fast resetting saccades.[1]

Gaze-shifting movements align the fovea with objects of interest and include saccades, smooth pursuit, and vergence. Saccades are fast, ballistic movements that redirect gaze, occurring voluntarily or reflexively, with a typical initiation latency of about 200 ms; once initiated, they cannot be altered midflight. Smooth pursuit movements track moving objects to keep them on the fovea but generally cannot be produced without a moving stimulus. Vergence movements adjust the angle between the eyes to align them with targets at varying distances, forming part of the near reflex triad along with accommodation and pupillary constriction. These are the only movements in which the eyes receive different motor commands, consistent with Hering’s law of equal innervation, which states that all conjugate eye movements (saccades and pursuit) involve identical commands to both eyes.[2]

_References:_
 [1] Glimcher PW. Eye movement, control of (oculomotor control). In: Smelser NJ, Baltes PB, editors. International encyclopedia of the social & behavioral sciences. Oxford: Pergamon; 2001. p. 5205-8.
 [2] Purves D, Augustine GJ, Fitzpatrick D, Katz LC, LaMantia AS, McNamara JO, Williams SM. Types of eye movements and their functions. Neuroscience. 2001;20:361-90.

------
