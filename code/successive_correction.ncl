begin

  nsite = 20        ; number of sites
  nimf  = 5         ; number of intrinsic mode functions (IMFs) 
;---Open file
  a     = addfile ("../results/CRAM_ConvLSTM.nc", "r")
  b     = addfile ("../results/EEMD_LSTM.nc", "r")
  
  pred      = a->SLA_CRAM_ConvLSTM
  ytst      = a->ytst
  eemd_lstm = b->eemd_lstm

  dsizes_x = dimsizes(pred)
  ntim     = dsizes_x(0)
  nstep    = dsizes_x(1)
  nlat     = dsizes_x(2)
  nlon     = dsizes_x(3)

  site_lat =new((/nsite/),integer,-99)
  site_lon =new((/nsite/),integer,-99)

  Site = asciiread("site_lonlat.txt",-1,"float")
  Site_shape=reshape(Site,(/nsite,5/))
  
do nSITE=0,nsite-1
  site_lat(nSITE) = numeric2int( Site_shape(nSITE,1),0 )
  site_lon(nSITE) = numeric2int( Site_shape(nSITE,0),0 )
end do

; ==============================================================
; successive correction
; ==============================================================
  HMnet_forecast = pred
  W=new((/1/),float,-999.9)

  Lx = 2
  Ly = 2
  Lf = 0.1

  do s = 0,nsite-1

  begi=site_lat(s)-Lx
  endi=site_lat(s)+Lx
  begj=site_lon(s)-Ly*2
  endj=site_lon(s)+Ly*2

  if ( begi .lt. 0 ) then
     begi = 0
  end if
  if ( begj .lt. 0 ) then
     begj = 0
  end if
  if ( endi .ge. nlat ) then
     endi = nlat-1
  end if
  if ( endj .ge. nlon ) then
     endj = nlon-1
  end if
  if ( begi .ge. nlat ) then
     begi = nlat-1
  end if
  if ( begj .ge. nlon ) then
     begj = nlon-1
  end if

  do i = begi, endi
  do j = begj, endj
  do m = 0, ntim-1
  do n = 0, nstep-1
    dt=HMnet_forecast(m,n,i,j)-HMnet_forecast(m,n,site_lat(s),site_lon(s))
    if(any(ismissing(dt))) then
      eemd_dt = 0.0
    else
      W = -(((i-site_lat(s))/Lx)^2)-(((j-site_lon(s))/Ly)^2)-((dt/Lf)^2)
      eemd_dt = (eemd_lstm(s,m,n)-HMnet_forecast(m,n,i,j))*2.71828^(W)
    end if
    HMnet_forecast(m,n,i,j)   = HMnet_forecast(m,n,i,j) + eemd_dt
  end do
  end do
  end do
  end do

  end do


  printVarSummary(HMnet_forecast)

  system("/bin/rm -f HMnet.nc")   ; remove any pre-existing file
  ncdf   = addfile("HMnet.nc" ,"c")  ; open output netCDF file
  filedimdef(ncdf,"time",-1,True)
  ncdf ->HMnet_SLA = HMnet_forecast


; ======= RMSE ========
          rmsd = dim_rmsd_n_Wrap(ytst,pred,0)  
;  print(rmsd)
          rmsd2 = dim_rmsd_n_Wrap(ytst,HMnet_forecast,0)  
;  print(rmsd2)
    rmsd = rmsd * 100 
    rmsd2 = rmsd2 * 100 
      ytst_min=rmsd(:,0,0)
      ytst_max=rmsd(:,0,0)
      ytst_ave=rmsd(:,0,0)
      ytst_ave_eemd=rmsd(:,0,0)

  do i = 0, nstep-1
    ytst_ave(i) = wgt_areaave(rmsd(i,:,:),1.0,1.0,0)
    ytst_ave_eemd(i) = wgt_areaave(rmsd2(i,:,:),1.0,1.0,0)
    ytst_min(i) = min(rmsd(i,:,:))
    ytst_max(i) = max(rmsd(i,:,:))
    ;ytst_max(i) = dim_max_n_Wrap(rmsd(i,:,:))
    ;ytst_min(i) = dim_min_n_Wrap(rmsd(i,:,:))
    printMinMax (rmsd(i,:,:),False)
  end do
print (ytst_ave)
print (ytst_ave_eemd)

end