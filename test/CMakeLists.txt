FILE(GLOB luasrc *.lua)
ADD_TORCH_PACKAGE(rnn/test "${src}" "${luasrc}")
