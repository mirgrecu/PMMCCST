subroutine grid_tb(tc_gmi_S1, lon_S1, lat_S1, lon, lat, &
    tc_resampled,ndpr, nray, ngmi, nfov, nchan)
    !use linear_interpolation_module
    use iso_fortran_env, only: real64
    integer :: ndpr, nray, ngmi, nfov, nchan
    real :: tc_gmi_S1(ngmi,nfov,nchan), lon_S1(ngmi,nfov), lat_S1(ngmi,nfov)
    real :: lon(ndpr,nray), lat(ndpr,nray)
    real(kind=real64) :: gmi_lon_lat_points(ngmi*nfov*2)
    real(kind=real64) :: dpr_lon_lat_points(ndpr*nray*2)
    real, intent(out) :: tc_resampled(ndpr,nray,nchan)
    integer :: closest_points(ndpr*nray*4)
    real(kind=real64) :: dist(ndpr*nray*4)
    integer :: i, j, k, ngmi_points, ndpr_points
    integer :: iclosest(4)
    real(kind=real64) :: tc_resampled_temp(nchan)
    real(kind=real64) :: dist_temp(4)
    real(kind=real64) :: weight, weight_sum
    do i=1,ngmi
        do j=1,nfov
            gmi_lon_lat_points((i-1)*nfov*2+(j-1)*2+1) = lon_S1(i,j)
            gmi_lon_lat_points((i-1)*nfov*2+(j-1)*2+2) = lat_S1(i,j)
        end do
    end do
    do i=1,ndpr
        do j=1,nray
            dpr_lon_lat_points((i-1)*nray*2+(j-1)*2+1) = lon(i,j)
            dpr_lon_lat_points((i-1)*nray*2+(j-1)*2+2) = lat(i,j)
        end do
    end do
    ngmi_points=ngmi*nfov
    ndpr_points=ndpr*nray
    call find_nearest(gmi_lon_lat_points, dpr_lon_lat_points, ngmi, ndpr, &
    closest_points, dist)
    do i=1,ndpr
        do j=1,nray
            iclosest=closest_points((i-1)*nray*4+(j-1)*4+1: (i-1)*nray*4+(j-1)*4+4)
            dist_temp=dist((i-1)*nray*4+(j-1)*4+1: (i-1)*nray*4+(j-1)*4+4)
            tc_resampled(i,j,:) = 0.0
            weight_sum=0
            do k=1,4
                igmi = iclosest(k)/nfov
                jgmi = mod(iclosest(k),nfov)
                weight=1/(dist_temp(k)**2+0.01)
                tc_resampled(i,j,:) = tc_resampled(i,j,:) + &
                tc_gmi_S1(igmi,jgmi,:)*weight
                weight_sum=weight_sum+weight
            enddo   
            tc_resampled(i,j,:) = tc_resampled(i,j,:)/weight_sum
        end do
    end do

end subroutine grid_tb