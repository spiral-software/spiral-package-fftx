#  specify cuda_required as true
#  specify cuda_target_name

cuda_required := true;
cuda_target_name := "linux-cuda";

#  set some defaults...
SpiralDefaults.profile := spiral.profiler.default_profiles.linux_x86_gcc;
SpiralDefaults.target := rec(name := "linux-cuda");
LocalConfig.osinfo := SupportedOSs.Linux64;
LocalConfig.compilerinfo := SupportedCompilers.NvidiaCuda;
