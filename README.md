# Timings

| Test | Previous | Latest | Improvement | Diff |
| :--- | :---: | :---: | ---: | ---: |
| test_plane_wave_propagation (tests.test_yee3d.TestYee3D.test_plane_wave_propagation) | 0.174 s | 69.975 s | x402.953 | +40195.30% |
| test_normalized_vectors (tests.test_examples.TestExamples.test_normalized_vectors) | 9.959 s | 15.303 s | x1.537 | +53.66% |
| test_fft_probes (tests.test_examples.TestExamples.test_fft_probes) | inf s | 12.722 s | xnan | +nan% |
| test_simple_propagation (tests.test_examples.TestExamples.test_simple_propagation) | 8.611 s | 11.341 s | x1.317 | +31.70% |
| test_probes (tests.test_examples.TestExamples.test_probes) | inf s | 9.371 s | xnan | +nan% |
| test_source_comparison (tests.test_examples.TestExamples.test_source_comparison) | 4.854 s | 6.907 s | x1.423 | +42.30% |
| test_coil_addition (tests.test_yee3d.TestYee3D.test_coil_addition) | 0.263 s | 2.700 s | x10.268 | +926.76% |
| test_plane_wave_dispersion (tests.test_dispersion.TestDispersion.test_plane_wave_dispersion) | 0.571 s | 0.810 s | x1.417 | +41.65% |
| test_skin_depth (tests.test_yee_skin_depth.TestYeeSkinDepth.test_skin_depth) | 0.065 s | 0.072 s | x1.099 | +9.86% |
| test_plane_wave (tests.test_yee_plane_wave_3d.TestYeePlaneWave3D.test_plane_wave) | 0.051 s | 0.061 s | x1.203 | +20.30% |
| test_step_stability (tests.test_yee3d.TestYee3D.test_step_stability) | 0.003 s | 0.003 s | x0.987 | -1.31% |
| test_cache (tests.test_cache.TestCache.test_cache) | 0.002 s | 0.003 s | x1.293 | +29.27% |
| test_initialization (tests.test_yee3d.TestYee3D.test_initialization) | 0.002 s | 0.002 s | x1.168 | +16.84% |
| test_pml_initialization (tests.test_yee3d.TestYee3D.test_pml_initialization) | 0.002 s | 0.002 s | x1.166 | +16.63% |
| test_set_materials (tests.test_yee3d.TestYee3D.test_set_materials) | 0.001 s | 0.001 s | x1.005 | +0.47% |

# Results

## demo_fft_probes

### fft_probe_comparison.png

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/demo_fft_probes/fft_probe_comparison.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/demo_fft_probes/fft_probe_comparison.png)|

### fft_probe_power.png

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/demo_fft_probes/fft_probe_power.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/demo_fft_probes/fft_probe_power.png)|

### fft_probe_signals.png

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/demo_fft_probes/fft_probe_signals.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/demo_fft_probes/fft_probe_signals.png)|

### fft_probe_spectra.png

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/demo_fft_probes/fft_probe_spectra.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/demo_fft_probes/fft_probe_spectra.png)|

## normalized_demo

### comparison_standard_vs_normalized.png

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/normalized_demo/comparison_standard_vs_normalized.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/normalized_demo/comparison_standard_vs_normalized.png)|

### normalized_different_colormaps.png

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/normalized_demo/normalized_different_colormaps.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/normalized_demo/normalized_different_colormaps.png)|

### normalized_different_densities.png

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/normalized_demo/normalized_different_densities.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/normalized_demo/normalized_different_densities.png)|

### normalized_optimal.png

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/normalized_demo/normalized_optimal.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/normalized_demo/normalized_optimal.png)|

## demo_probes

### probe_example_corners.png

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/demo_probes/probe_example_corners.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/demo_probes/probe_example_corners.png)|

### probe_example_line.png

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/demo_probes/probe_example_line.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/demo_probes/probe_example_line.png)|

### probe_example_source.png

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/demo_probes/probe_example_source.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/demo_probes/probe_example_source.png)|

## demo_simple

### propagation_comparaison.png

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/demo_simple/propagation_comparaison.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/demo_simple/propagation_comparaison.png)|

### propagation_streamlines.png

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/demo_simple/propagation_streamlines.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/demo_simple/propagation_streamlines.png)|

### propagation_vecteurs_normalises.png

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/demo_simple/propagation_vecteurs_normalises.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/demo_simple/propagation_vecteurs_normalises.png)|

## source_comparison

### continuous_vs_pulse.png

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/source_comparison/continuous_vs_pulse.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/source_comparison/continuous_vs_pulse.png)|

### signal_comparison.png

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/source_comparison/signal_comparison.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/source_comparison/signal_comparison.png)|

## anim_01_dielectric

### refraction_animation.gif

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/anim_01_dielectric/refraction_animation.gif)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/anim_01_dielectric/refraction_animation.gif)|

### refraction_animation.mp4

#### Before

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/anim_01_dielectric/refraction_animation.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/anim_01_dielectric/refraction_animation.mp4

## anim_01_vectors

### frame_0000.png

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/anim_01_vectors/frame_0000.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/anim_01_vectors/frame_0000.png)|

### frame_0001.png

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/anim_01_vectors/frame_0001.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/anim_01_vectors/frame_0001.png)|

### frame_0002.png

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/anim_01_vectors/frame_0002.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/anim_01_vectors/frame_0002.png)|

### frame_0003.png

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/anim_01_vectors/frame_0003.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/anim_01_vectors/frame_0003.png)|

### frame_0004.png

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/anim_01_vectors/frame_0004.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/anim_01_vectors/frame_0004.png)|

### frame_0005.png

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/anim_01_vectors/frame_0005.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/anim_01_vectors/frame_0005.png)|

### frame_0006.png

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/anim_01_vectors/frame_0006.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/anim_01_vectors/frame_0006.png)|

### frame_0007.png

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/anim_01_vectors/frame_0007.png)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/anim_01_vectors/frame_0007.png)|

### refraction_animation.gif

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/anim_01_vectors/refraction_animation.gif)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/anim_01_vectors/refraction_animation.gif)|

### refraction_vectors.gif

| Before | After |
| --- | --- |
| ![Before](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/anim_01_vectors/refraction_vectors.gif)| ![After](https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/anim_01_vectors/refraction_vectors.gif)|

### refraction_vectors.mp4

#### Before

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/anim_01_vectors/refraction_vectors.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/anim_01_vectors/refraction_vectors.mp4

## anim_02_metal

### metal_reflection.mp4

#### Before

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/anim_02_metal/metal_reflection.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/anim_02_metal/metal_reflection.mp4

## anim_02_vectors

### reflexion_metal_vectors.mp4

#### Before

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/anim_02_vectors/reflexion_metal_vectors.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/anim_02_vectors/reflexion_metal_vectors.mp4

## anim_03_lossy

### lossy_medium.mp4

#### Before

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/anim_03_lossy/lossy_medium.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/anim_03_lossy/lossy_medium.mp4

## anim_03_vectors

### attenuation_vectors.mp4

#### Before

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/anim_03_vectors/attenuation_vectors.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/anim_03_vectors/attenuation_vectors.mp4

## anim_04_cavity

### cavity_resonance.mp4

#### Before

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/anim_04_cavity/cavity_resonance.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/anim_04_cavity/cavity_resonance.mp4

## anim_04_vectors

### cavite_resonante_vectors.mp4

#### Before

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/anim_04_vectors/cavite_resonante_vectors.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/anim_04_vectors/cavite_resonante_vectors.mp4

## anim_05_multilayer

### multilayer.mp4

#### Before

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/anim_05_multilayer/multilayer.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/anim_05_multilayer/multilayer.mp4

## anim_05_vectors

### multicouche_vectors.mp4

#### Before

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/a6174fc8b107f691cfbe5abc83ba8fee0ce36778/examples/results/anim_05_vectors/multicouche_vectors.mp4

#### After

https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/f948bc85ce292449ff61d456f00a5fa0792c496e/examples/results/anim_05_vectors/multicouche_vectors.mp4

