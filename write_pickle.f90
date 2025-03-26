subroutine write_pickle_land(tc, sfc_type, sk_temp, oe_wvp, near_sfc_precip, xenc, xenc_prec, xenv_enc)
    real :: tc(2,150,49,13)
    real :: sfc_type(2,150,49)
    real :: sk_temp(2,150,49)
    real :: oe_wvp(2,150,49)
    real :: near_sfc_precip(2,150,49)
    real :: xenc(2,150,49,4)
    real :: xenc_prec(2,150,49,6)
    real :: xenv_enc(2,150,49,4)
    character(*), parameter :: bin_fname='GMI_ONNX_Models/norm_param_150_land.bin'
  
    open(10,file=bin_fname,form='unformatted',status='unknown')
    write(10)tc
    write(10)sfc_type
    write(10)sk_temp
    write(10)oe_wvp
    write(10)near_sfc_precip
    write(10)xenc
    write(10)xenc_prec
    write(10)xenv_enc
    close(10)
end subroutine write_pickle_land    

subroutine write_pickle_ocean(tc, sfc_type, sk_temp, oe_wvp, near_sfc_precip, xenc, xenc_prec, xenv_enc)
    real :: tc(2,150,49,13)
    real :: sfc_type(2,150,49)
    real :: sk_temp(2,150,49)
    real :: oe_wvp(2,150,49)
    real :: near_sfc_precip(2,150,49)
    real :: xenc(2,150,49,4)
    real :: xenc_prec(2,150,49,6)
    real :: xenv_enc(2,150,49,4)
    character(*), parameter :: bin_fname='GMI_ONNX_Models/norm_param_150_ocean.bin'
  
    open(10,file=bin_fname,form='unformatted',status='unknown')
    write(10)tc
    write(10)sfc_type
    write(10)sk_temp
    write(10)oe_wvp
    write(10)near_sfc_precip
    write(10)xenc
    write(10)xenc_prec
    write(10)xenv_enc
    close(10)
end subroutine write_pickle_ocean