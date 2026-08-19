# Painting photographs

**These are committed.** They are the source data for every coordinate in the
piece, and Stage 1a is not reproducible without them.

Expected names:

    panel_a_photo.jpg    the more crowded canvas -- painted from the photograph
                         of real classmates
    panel_b_photo.jpg    the sparser canvas -- painted from the AI-generated image

Shoot notes that matter for Stage 1a: the whole canvas in frame with a margin of
background on all four sides, as square-on as practical, even light, no specular
hotspot on the gloss. Perspective is corrected by `stage1/rectify.py`, but a
grazing angle costs real resolution on the far edge, and a highlight sitting on
a face costs that face.

Highest resolution available, please. Detection runs on the rectified raster at
120 px/cm (4800 x 3600), so anything below roughly 4000 px on the long edge is
being upsampled and the small background faces will suffer for it.
