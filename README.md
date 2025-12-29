# Timings

| Test | Previous | Latest | Improvement | Diff |
| :--- | :---: | :---: | ---: | ---: |
<<<<<<< HEAD
| test_03_lossy_medium (tests.test_examples.TestExamples.test_03_lossy_medium) | inf s | 90.652 s | x0.000 | -inf% |
| test_05_layered_materials (tests.test_examples.TestExamples.test_05_layered_materials) | inf s | 86.865 s | x0.000 | -inf% |
| test_02_metal_reflection (tests.test_examples.TestExamples.test_02_metal_reflection) | inf s | 79.310 s | x0.000 | -inf% |
| test_04_dielectric_cavity (tests.test_examples.TestExamples.test_04_dielectric_cavity) | inf s | 78.766 s | x0.000 | -inf% |
| test_plane_wave_propagation (tests.test_yee3d.TestYee3D.test_plane_wave_propagation) | inf s | 42.317 s | x0.000 | -inf% |
| test_01_dielectric_refraction (tests.test_examples.TestExamples.test_01_dielectric_refraction) | inf s | 41.802 s | x0.000 | -inf% |
| test_normalized_vectors (tests.test_examples.TestExamples.test_normalized_vectors) | inf s | 9.562 s | x0.000 | -inf% |
| test_simple_propagation (tests.test_examples.TestExamples.test_simple_propagation) | inf s | 8.373 s | x0.000 | -inf% |
| test_fft_probes (tests.test_examples.TestExamples.test_fft_probes) | inf s | 7.801 s | x0.000 | -inf% |
| test_probes (tests.test_examples.TestExamples.test_probes) | inf s | 6.292 s | x0.000 | -inf% |
| test_source_comparison (tests.test_examples.TestExamples.test_source_comparison) | inf s | 5.103 s | x0.000 | -inf% |
| test_coil_addition (tests.test_yee3d.TestYee3D.test_coil_addition) | inf s | 2.131 s | x0.000 | -inf% |
| test_plane_wave_dispersion (tests.test_dispersion.TestDispersion.test_plane_wave_dispersion) | inf s | 0.683 s | x0.000 | -inf% |
| test_skin_depth (tests.test_yee_skin_depth.TestYeeSkinDepth.test_skin_depth) | inf s | 0.064 s | x0.000 | -inf% |
| test_plane_wave (tests.test_yee_plane_wave_3d.TestYeePlaneWave3D.test_plane_wave) | inf s | 0.050 s | x0.000 | -inf% |
| test_step_stability (tests.test_yee3d.TestYee3D.test_step_stability) | inf s | 0.003 s | x0.000 | -inf% |
| test_cache (tests.test_cache.TestCache.test_cache) | inf s | 0.002 s | x0.000 | -inf% |
| test_pml_initialization (tests.test_yee3d.TestYee3D.test_pml_initialization) | inf s | 0.002 s | x0.000 | -inf% |
| test_initialization (tests.test_yee3d.TestYee3D.test_initialization) | inf s | 0.002 s | x0.000 | -inf% |
| test_set_materials (tests.test_yee3d.TestYee3D.test_set_materials) | inf s | 0.001 s | x0.000 | -inf% |
=======
| test_03_vector_lossy (tests.test_examples.TestExamples.test_03_vector_lossy) | 273.539 s | :fast_forward: | = | = |
| test_04_vector_cavity (tests.test_examples.TestExamples.test_04_vector_cavity) | 262.682 s | :fast_forward: | = | = |
| test_01_vector_refraction (tests.test_examples.TestExamples.test_01_vector_refraction) | 188.845 s | :fast_forward: | = | = |
| test_02_vector_metal (tests.test_examples.TestExamples.test_02_vector_metal) | 129.177 s | :fast_forward: | = | = |
| test_05_vector_multilayer (tests.test_examples.TestExamples.test_05_vector_multilayer) | 108.950 s | :fast_forward: | = | = |
| test_plane_wave_propagation (tests.test_yee3d.TestYee3D.test_plane_wave_propagation) | 65.966 s | 60.370 s | x0.915 | -6.35% |
| test_03_lossy_medium (tests.test_examples.TestExamples.test_03_lossy_medium) | 56.804 s | :fast_forward: | = | = |
| test_05_layered_materials (tests.test_examples.TestExamples.test_05_layered_materials) | 54.728 s | :fast_forward: | = | = |
| test_02_metal_reflection (tests.test_examples.TestExamples.test_02_metal_reflection) | 51.817 s | :fast_forward: | = | = |
| test_04_dielectric_cavity (tests.test_examples.TestExamples.test_04_dielectric_cavity) | 50.603 s | :fast_forward: | = | = |
| test_01_dielectric_refraction (tests.test_examples.TestExamples.test_01_dielectric_refraction) | 35.577 s | :fast_forward: | = | = |
| test_normalized_vectors (tests.test_examples.TestExamples.test_normalized_vectors) | 6.546 s | 6.994 s | x1.068 | +0.51% |
| test_fft_probes (tests.test_examples.TestExamples.test_fft_probes) | 5.247 s | 5.480 s | x1.045 | +0.27% |
| test_simple_propagation (tests.test_examples.TestExamples.test_simple_propagation) | 5.073 s | 5.389 s | x1.062 | +0.36% |
| test_probes (tests.test_examples.TestExamples.test_probes) | 4.156 s | 4.406 s | x1.060 | +0.28% |
| test_source_comparison (tests.test_examples.TestExamples.test_source_comparison) | 3.042 s | 3.083 s | x1.013 | +0.05% |
| test_coil_addition (tests.test_yee3d.TestYee3D.test_coil_addition) | 1.783 s | 1.892 s | x1.061 | +0.12% |
| test_plane_wave_dispersion (tests.test_dispersion.TestDispersion.test_plane_wave_dispersion) | 0.465 s | 0.456 s | x0.980 | -0.01% |
| test_skin_depth (tests.test_yee_skin_depth.TestYeeSkinDepth.test_skin_depth) | 0.041 s | 0.042 s | x1.009 | +0.00% |
| test_plane_wave (tests.test_yee_plane_wave_3d.TestYeePlaneWave3D.test_plane_wave) | 0.032 s | 0.031 s | x0.958 | -0.00% |
| test_step_stability (tests.test_yee3d.TestYee3D.test_step_stability) | 0.002 s | 0.002 s | x0.935 | -0.00% |
| test_pml_initialization (tests.test_yee3d.TestYee3D.test_pml_initialization) | 0.001 s | 0.001 s | x1.087 | +0.00% |
| test_initialization (tests.test_yee3d.TestYee3D.test_initialization) | 0.001 s | 0.001 s | x1.019 | +0.00% |
| test_set_materials (tests.test_yee3d.TestYee3D.test_set_materials) | 0.001 s | 0.001 s | x1.079 | +0.00% |
| test_cache (tests.test_cache.TestCache.test_cache) | 0.000 s | 0.000 s | x0.967 | -0.00% |
>>>>>>> 62780759b93553f828334f07185f4290834ca526

# Results

## anim_01_dielectric

### refraction_animation.gif

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/anim_01_dielectric/refraction_animation.gif)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/anim_01_dielectric/refraction_animation.gif)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/anim_01_dielectric/refraction_animation.gif)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/anim_01_dielectric/refraction_animation.gif)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

### refraction_animation.mp4

#### Before

<<<<<<< HEAD
https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/anim_01_dielectric/refraction_animation.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/anim_01_dielectric/refraction_animation.mp4
=======
https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/anim_01_dielectric/refraction_animation.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/anim_01_dielectric/refraction_animation.mp4
>>>>>>> 62780759b93553f828334f07185f4290834ca526

## anim_01_vectors

### frame_0000.png

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/anim_01_vectors/frame_0000.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/anim_01_vectors/frame_0000.png)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/anim_01_vectors/frame_0000.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/anim_01_vectors/frame_0000.png)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

### frame_0001.png

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/anim_01_vectors/frame_0001.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/anim_01_vectors/frame_0001.png)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/anim_01_vectors/frame_0001.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/anim_01_vectors/frame_0001.png)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

### frame_0002.png

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/anim_01_vectors/frame_0002.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/anim_01_vectors/frame_0002.png)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/anim_01_vectors/frame_0002.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/anim_01_vectors/frame_0002.png)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

### frame_0003.png

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/anim_01_vectors/frame_0003.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/anim_01_vectors/frame_0003.png)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/anim_01_vectors/frame_0003.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/anim_01_vectors/frame_0003.png)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

### frame_0004.png

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/anim_01_vectors/frame_0004.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/anim_01_vectors/frame_0004.png)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/anim_01_vectors/frame_0004.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/anim_01_vectors/frame_0004.png)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

### frame_0005.png

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/anim_01_vectors/frame_0005.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/anim_01_vectors/frame_0005.png)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/anim_01_vectors/frame_0005.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/anim_01_vectors/frame_0005.png)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

### frame_0006.png

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/anim_01_vectors/frame_0006.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/anim_01_vectors/frame_0006.png)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/anim_01_vectors/frame_0006.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/anim_01_vectors/frame_0006.png)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

### frame_0007.png

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/anim_01_vectors/frame_0007.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/anim_01_vectors/frame_0007.png)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/anim_01_vectors/frame_0007.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/anim_01_vectors/frame_0007.png)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

### refraction_animation.gif

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/anim_01_vectors/refraction_animation.gif)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/anim_01_vectors/refraction_animation.gif)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/anim_01_vectors/refraction_animation.gif)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/anim_01_vectors/refraction_animation.gif)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

### refraction_vectors.gif

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/anim_01_vectors/refraction_vectors.gif)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/anim_01_vectors/refraction_vectors.gif)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/anim_01_vectors/refraction_vectors.gif)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/anim_01_vectors/refraction_vectors.gif)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

### refraction_vectors.mp4

#### Before

<<<<<<< HEAD
https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/anim_01_vectors/refraction_vectors.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/anim_01_vectors/refraction_vectors.mp4
=======
https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/anim_01_vectors/refraction_vectors.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/anim_01_vectors/refraction_vectors.mp4
>>>>>>> 62780759b93553f828334f07185f4290834ca526

## anim_02_metal

### metal_reflection.mp4

#### Before

<<<<<<< HEAD
https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/anim_02_metal/metal_reflection.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/anim_02_metal/metal_reflection.mp4
=======
https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/anim_02_metal/metal_reflection.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/anim_02_metal/metal_reflection.mp4
>>>>>>> 62780759b93553f828334f07185f4290834ca526

## anim_02_vectors

### reflexion_metal_vectors.mp4

#### Before

<<<<<<< HEAD
https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/anim_02_vectors/reflexion_metal_vectors.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/anim_02_vectors/reflexion_metal_vectors.mp4
=======
https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/anim_02_vectors/reflexion_metal_vectors.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/anim_02_vectors/reflexion_metal_vectors.mp4
>>>>>>> 62780759b93553f828334f07185f4290834ca526

## anim_03_lossy

### lossy_medium.mp4

#### Before

<<<<<<< HEAD
https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/anim_03_lossy/lossy_medium.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/anim_03_lossy/lossy_medium.mp4
=======
https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/anim_03_lossy/lossy_medium.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/anim_03_lossy/lossy_medium.mp4
>>>>>>> 62780759b93553f828334f07185f4290834ca526

## anim_03_vectors

### attenuation_vectors.mp4

#### Before

<<<<<<< HEAD
https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/anim_03_vectors/attenuation_vectors.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/anim_03_vectors/attenuation_vectors.mp4
=======
https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/anim_03_vectors/attenuation_vectors.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/anim_03_vectors/attenuation_vectors.mp4
>>>>>>> 62780759b93553f828334f07185f4290834ca526

## anim_04_cavity

### cavity_resonance.mp4

#### Before

<<<<<<< HEAD
https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/anim_04_cavity/cavity_resonance.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/anim_04_cavity/cavity_resonance.mp4
=======
https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/anim_04_cavity/cavity_resonance.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/anim_04_cavity/cavity_resonance.mp4
>>>>>>> 62780759b93553f828334f07185f4290834ca526

## anim_04_vectors

### cavite_resonante_vectors.mp4

#### Before

<<<<<<< HEAD
https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/anim_04_vectors/cavite_resonante_vectors.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/anim_04_vectors/cavite_resonante_vectors.mp4
=======
https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/anim_04_vectors/cavite_resonante_vectors.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/anim_04_vectors/cavite_resonante_vectors.mp4
>>>>>>> 62780759b93553f828334f07185f4290834ca526

## anim_05_multilayer

### multilayer.mp4

#### Before

<<<<<<< HEAD
https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/anim_05_multilayer/multilayer.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/anim_05_multilayer/multilayer.mp4
=======
https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/anim_05_multilayer/multilayer.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/anim_05_multilayer/multilayer.mp4
>>>>>>> 62780759b93553f828334f07185f4290834ca526

## anim_05_vectors

### multicouche_vectors.mp4

#### Before

<<<<<<< HEAD
https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/anim_05_vectors/multicouche_vectors.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/anim_05_vectors/multicouche_vectors.mp4
=======
https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/anim_05_vectors/multicouche_vectors.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/anim_05_vectors/multicouche_vectors.mp4
>>>>>>> 62780759b93553f828334f07185f4290834ca526

## demo_fft_probes

### fft_probe_comparison.png

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/demo_fft_probes/fft_probe_comparison.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/demo_fft_probes/fft_probe_comparison.png)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/demo_fft_probes/fft_probe_comparison.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/demo_fft_probes/fft_probe_comparison.png)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

### fft_probe_power.png

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/demo_fft_probes/fft_probe_power.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/demo_fft_probes/fft_probe_power.png)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/demo_fft_probes/fft_probe_power.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/demo_fft_probes/fft_probe_power.png)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

### fft_probe_signals.png

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/demo_fft_probes/fft_probe_signals.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/demo_fft_probes/fft_probe_signals.png)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/demo_fft_probes/fft_probe_signals.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/demo_fft_probes/fft_probe_signals.png)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

### fft_probe_spectra.png

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/demo_fft_probes/fft_probe_spectra.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/demo_fft_probes/fft_probe_spectra.png)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/demo_fft_probes/fft_probe_spectra.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/demo_fft_probes/fft_probe_spectra.png)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

## demo_probes

### probe_example_corners.png

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/demo_probes/probe_example_corners.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/demo_probes/probe_example_corners.png)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/demo_probes/probe_example_corners.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/demo_probes/probe_example_corners.png)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

### probe_example_line.png

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/demo_probes/probe_example_line.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/demo_probes/probe_example_line.png)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/demo_probes/probe_example_line.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/demo_probes/probe_example_line.png)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

### probe_example_source.png

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/demo_probes/probe_example_source.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/demo_probes/probe_example_source.png)|

### probe_source_data.npz

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/demo_probes/probe_source_data.npz)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/demo_probes/probe_source_data.npz)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/demo_probes/probe_example_source.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/demo_probes/probe_example_source.png)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

## demo_simple

### propagation_comparaison.png

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/demo_simple/propagation_comparaison.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/demo_simple/propagation_comparaison.png)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/demo_simple/propagation_comparaison.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/demo_simple/propagation_comparaison.png)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

### propagation_streamlines.png

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/demo_simple/propagation_streamlines.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/demo_simple/propagation_streamlines.png)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/demo_simple/propagation_streamlines.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/demo_simple/propagation_streamlines.png)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

### propagation_vecteurs_normalises.png

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/demo_simple/propagation_vecteurs_normalises.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/demo_simple/propagation_vecteurs_normalises.png)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/demo_simple/propagation_vecteurs_normalises.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/demo_simple/propagation_vecteurs_normalises.png)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

## normalized_demo

### comparison_standard_vs_normalized.png

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/normalized_demo/comparison_standard_vs_normalized.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/normalized_demo/comparison_standard_vs_normalized.png)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/normalized_demo/comparison_standard_vs_normalized.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/normalized_demo/comparison_standard_vs_normalized.png)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

### normalized_different_colormaps.png

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/normalized_demo/normalized_different_colormaps.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/normalized_demo/normalized_different_colormaps.png)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/normalized_demo/normalized_different_colormaps.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/normalized_demo/normalized_different_colormaps.png)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

### normalized_different_densities.png

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/normalized_demo/normalized_different_densities.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/normalized_demo/normalized_different_densities.png)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/normalized_demo/normalized_different_densities.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/normalized_demo/normalized_different_densities.png)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

### normalized_optimal.png

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/normalized_demo/normalized_optimal.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/normalized_demo/normalized_optimal.png)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/normalized_demo/normalized_optimal.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/normalized_demo/normalized_optimal.png)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

## source_comparison

### continuous_vs_pulse.png

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/source_comparison/continuous_vs_pulse.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/source_comparison/continuous_vs_pulse.png)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/source_comparison/continuous_vs_pulse.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/source_comparison/continuous_vs_pulse.png)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

### signal_comparison.png

| Before | After |
| --- | --- |
<<<<<<< HEAD
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a873a12d9b553353860981dc27aa5e0142c4640e/examples/results/source_comparison/signal_comparison.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/20d7b76358a27c312f795ebc1f57573c7f90bd62/examples/results/source_comparison/signal_comparison.png)|
=======
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/acc0288ee6e6e7d0c2625a39c1d89dea5c82edbb/examples/results/source_comparison/signal_comparison.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/e21f033f64cec042ba525a79fb110223ddaa4632/examples/results/source_comparison/signal_comparison.png)|
>>>>>>> 62780759b93553f828334f07185f4290834ca526

