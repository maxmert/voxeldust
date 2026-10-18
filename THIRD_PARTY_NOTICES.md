# Third-party notices

Voxeldust's water drawing ports methods, and may port code, from the projects below (ruling B106). Their licences ask
that these notices travel with the code. No texture, sound or other asset from them is shipped.

## bevy-aqua
- Source: https://github.com/sayhisam1/bevy-aqua
- Licence: MIT OR Apache-2.0 (the licence files `LICENSE-MIT` and `LICENSE-APACHE` of that repository).
- Used for: the water's lighting (the Atlas BSDF with GGX), absorption and scattering by depth, LEADR roughness from the
  slope variance, foam methods, the motion of moving water.

## Crest (via bevy-aqua's ATTRIBUTION.md)
- Source: https://github.com/wave-harmonic/crest
- Copyright (c) 2019 Wave Harmonic and contributors. Licence: MIT.
- Used for: the analytic wave methods bevy-aqua credits to it. Its textures (WaveNormals.png, Foam2.png) are not shipped.

## GodotOceanWaves
- Source: https://github.com/2Retr0/GodotOceanWaves
- Copyright (c) 2024 Ethan Truong. Licence: MIT.
- Used for: the TMA wave spectrum with Hasselmann's directional spread and the finite-depth dispersion.

## Published methods (no code)
- Jeschke, Skřivan, Müller-Fischer, Chentanez, Macklin and Wojtan (2018), *Water Surface Wavelets*, ACM Transactions on
  Graphics 37(4): the profile buffers — the waves as 1D profiles along fixed directions.

## The MIT licence text
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
