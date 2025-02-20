subroutine init_onnx_f90()
    call init_onnx_runtime()
    print*, 'here'
end subroutine init_onnx_f90
!call_onnx_(float *input_data, int *lengths_data, float *output_data, int *batch_size, int *seq_len, int *input_size, int *output_size)

!void call_onnx_(float *input_data, int *lengths_data, float *output_data, int *batch_size, int *seq_len, int *input_size, int *output_size, int *im, float !*input_data2, float *output_data2, int *output_2_size)

subroutine onnx_retrieval_ku_f90(z_ku_meas, p_type, bin_nodes, n_scans, clut_depth, onnx_precip_rate, &
     onnx_dm, &
     near_surf_onnx_precip_rate,pia_output,xlon,xlat,n_batch, n_seq, n_input,n_output,n_dim_2, n_comp_2)
    implicit none
    integer :: n_scans, n_dim_2, n_comp_2
    real :: z_ku_meas(88,49,n_scans)
    real, intent(out) :: pia_output(n_dim_2*(n_comp_2+2),49,n_scans)
    integer :: p_type(49,n_scans) , bin_nodes(5,49,n_scans)
    real, intent(out) :: onnx_precip_rate(88,49,n_scans)
    real, intent(out) :: onnx_dm(88, 49,n_scans)
    real :: clut_depth(1,49,n_scans)
    integer :: n_batch, n_seq, n_input, n_output
    real :: input_1_data(n_input,n_seq,n_batch), input_2_data(1,n_batch)
    integer :: actual_seq_len(n_batch)
    real :: output_1_data(n_output,n_seq,n_batch), output_2_data(n_dim_2*(n_comp_2+2),n_batch)
    integer :: i, j, k, ic
    real :: z_ku_scaled, bin_scaled
    real, intent(out) :: near_surf_onnx_precip_rate(49,n_scans)
    integer :: n_points=1
    real :: xlon(49,n_scans), xlat(49,n_scans), wfractPix
    integer :: im
    ic=1
    !print*, n_points,n_batch,n_seq,n_input,n_output
    !print*, n_scans
    onnx_dm=0
    onnx_precip_rate=0
    near_surf_onnx_precip_rate=0
    print*, 'nscans=',n_scans
    do i=1,n_scans
        do j=1,49
           if (p_type(j,i) > 0) then
              !print*, p_type(j,i), bin_nodes(:,j,i)
                do k=bin_nodes(1,j,i),bin_nodes(5,j,i)
                    z_ku_scaled=z_ku_meas(k,j,i)
                    if (z_ku_scaled<0) z_ku_scaled=0
                    z_ku_scaled=(z_ku_scaled-12)/8
                    bin_scaled=(k-bin_nodes(3,j,i))/8.0
                    if (k+1-bin_nodes(1,j,i) <= n_seq) then
                        input_1_data(1,k+1-bin_nodes(1,j,i),ic)=z_ku_scaled
                        input_1_data(2,k+1-bin_nodes(1,j,i),ic)=bin_scaled
                    end if
                end do
                actual_seq_len(ic)=bin_nodes(5,j,i)-bin_nodes(1,j,i)+1
                input_2_data(1,ic)=clut_depth(1,j,i)
                !call getwfraction(xlat(j,i),&
                !     xlon(j,i),wfractPix)
                im=0
                if(p_type(j,i)/100==2) im=1
                wfractPix=100
                !if(wfractPix<50) im=im+2
                !print*, i, j, ic, actual_seq_len(ic)
                if (actual_seq_len(ic)>1) then
!void call_onnx_(float *input_data, int *lengths_data, float *output_data, int *batch_size, int *seq_len, int *input_size, int *output_size, int *im, float !*input_data2, float *output_data2, int *output_2_size)
                                       
                    call call_onnx(input_1_data, actual_seq_len, output_1_data, n_points, & 
                    n_seq, n_input, n_output,im, input_2_data, output_2_data, n_dim_2*(n_comp_2+2))
                    if (actual_seq_len(ic) > 20) then
                        !print*, output_2_data(:,ic)
                    end if
                endif
                do k=bin_nodes(1,j,i),bin_nodes(5,j,i)
                    onnx_precip_rate(k,j,i)=0.1*(10**output_1_data(1,k+1-bin_nodes(1,j,i),ic)-1)
                    onnx_dm(k,j,i)=output_1_data(2,k+1-bin_nodes(1,j,i),ic)
                    if(k==bin_nodes(5,j,i)-1) then
                        near_surf_onnx_precip_rate(j,i)=0.1*(10**output_1_data(1,k+1-bin_nodes(1,j,i),ic)-1)
                    end if
                end do
                do k=1,n_dim_2*(n_comp_2+2)
                    pia_output(k,j,i)=output_2_data(k,ic)
                end do
            end if
            
        end do
    end do
    print*, n_input, n_seq, n_batch, n_output
    print*, ic
    
end subroutine onnx_retrieval_ku_f90


