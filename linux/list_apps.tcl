setws "D:/Users/u/tmac-zynq-fpga/linux/boot"
platform create -name "tmp_platform" -hw "D:/Users/u/tmac-zynq-fpga/linux/boot/matmul_bd.xsa" -proc ps7_cortexa9_0 -os standalone
app create -name "test_app" -platform "tmp_platform" -template "Empty Application(C)" -proc ps7_cortexa9_0 -os standalone
