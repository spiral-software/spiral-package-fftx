#  specify cuda_required as true
#  specify cuda_target_name

cuda_required := true;
cuda_target_name := "win-x64-cuda";

#  set some defaults...
UseNVCC := true;
SpiralDefaults.profile := spiral.profiler.default_profiles.win_x64_cuda;
SpiralDefaults.target := rec(name := "win-x64-cuda");
LocalConfig.compilerinfo := SupportedCompilers.NvidiaCuda;
