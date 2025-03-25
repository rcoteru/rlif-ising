# Refractive Leaky Integrate-and-Fire Ising model
[![CI](https://github.com/rcoteru/RefractiveIsing/actions/workflows/CI.yml/badge.svg)](https://github.com/rcoteru/RefractiveIsing/actions/workflows/CI.yml)


Code to reproduce the figures in the article "_Ising Model of Refractive Leaky Integrate-and-Fire Neurons_" by [Raúl Coterillo](https://orcid.org/0000-0001-7567-3646) and [Miguel Aguilera](https://orcid.org/0000-0002-3366-4706).


## Reproducibility

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "BioIsing"
```
which auto-activate the project and enable local path handling from DrWatson.