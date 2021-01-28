# get “system default” as defined by SPIRAL startup scripts
LocalConfig.defaultConf();

# pick from a number of default configs
LocalConfig.supportedConfs.confGPU();
LocalConfig.supportedConfs.confMultiGPU();
LocalConfig.supportedConfs.confSclarCPU();
LocalConfig.supportedConfs.confOMPVMX();
…

# guru interface by configuring confs
LocalConfig.defaultConf(rec(useCPU := true, useGPU := false, useP9 := true, useOpenMP := true));
LocalConfig.defaultConf(rec(useCPU := false, useGPU := true, useMultiGPU := false));
LocalConfig.supportedConfs.confOMPVMX(rec(threads := 4, OMPver := “4.1.2”));
