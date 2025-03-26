module scaler_2d_land
    real :: tc(2,150,49,13)
    real :: sfc_type(2,150,49)
    real :: sk_temp(2,150,49)
    real :: oe_wvp(2,150,49)
    real :: near_sfc_precip(2,150,49)
    real :: xenc(2,150,49,4)
    real :: xenc_prec(2,150,49,6)
    real :: xenv_enc(2,150,49,4)
end module scaler_2d_land
subroutine read_norm_land()
  use scaler_2d_land
  character(*), parameter :: bin_fname='GMI_ONNX_Models/norm_param_150_land.bin'
  
  open(10,file=bin_fname,form='unformatted',status='old')
  read(10)tc
  read(10)sfc_type
  read(10)sk_temp
  read(10)oe_wvp
  read(10)near_sfc_precip
  read(10)xenc
  read(10)xenc_prec
  read(10)xenv_enc
  close(10)
end subroutine read_norm_land
