#! /bin/csh
#
# --- Purpose: 
# ---   do successive correction of wind and temperature
# -----------------------------------------------------
#
# --- step 0 : Setting preparing
# ---  Notice: if date_time type is given, please do set: 
# ---          e.g, 
# ---          set date_time = 20100815_08 ( LST Time ! )
# ---                                YYYYMMDD_HH
# ---

   set current_path = /home/seafog/work/correction_soft
   set verify_path  = ${current_path}/data/verify
   set obs_path     = ${current_path}/data/obs
   set bg_path      = ${current_path}/data/bg

   # --- time and bg domain
   # ---
   set date_time = 20110919_00
   set lon_beg   = '40.0'
   set lon_end   = '160.0'
   set lat_beg   = '0.0'
   set lat_end   = '60.0'
   set res       = '0.25'
   
   # --- if_bogus_TC
   # ---
   set if_bogus = yes

   # --- if_bogus = yes, then set below
   # ---
   # --- TC_numbers must more than 1 (include), 
   # --- and less than 20.
   # ---
   set TC_num = 1
   set TC_lon = '130.5,144.1,'
   set TC_lat = '26.8,34.5,'
   set vmax   = '30.0,33.0,'
   set rmax   = '60.0,60.0,'
   set ratio  = '0.85,0.85,'
   set R50    = '50.0,50.0,'
   set R30    = '280.0,200.0,'
   set bogus_scheme = '3,3,'

   # --- varible to successive correction
   # --- var: u-wind, v-wind, temperature
   # --- num:   1/0    1/0        1/0
   # ---
   set var_select = '1,1,1,'

   # --- successive options
   # ---
   set window_time = '60,'
   set wind_radius = '100,80,30,'
   set temp_radius = '50,30,20,'
   set obs_radius  = '200,'
   set verify_result = yes

# =======================================
# ==== Stop your modifications here ! ===
# =======================================
# ---
# --- Preparing
# ---
  if( -e mask.nc ) then
  set clean_file = " mask.nc wsat_????????v7 ????????.sfc wrf_????????_??.nc wrf????????_??.nc_cfsr namelist.sc succ_*.gs msg_plot msg original_uv.nc successive_uv.nc "
  foreach del_file (${clean_file})
    if( -d ${del_file} || -l ${del_file} || -e ${del_file} ) then
      rm -rf ${del_file}
    endif
  end
  endif

# ---
# --- step 1 : create date_fime for obs_files
# ---

  cd $current_path
  # 
  set year = `echo $date_time | cut -c1-4`
  set mm  = `echo $date_time | cut -c5-6`
  set dd  = `echo $date_time | cut -c7-8`
  set hh  = `echo $date_time | cut -c10-11`
  # UTC2LST
  set yy_lst = `date +%y --date="$year$mm$dd $hh 8 hours"`
  set mm_lst = `date +%m --date="$year$mm$dd $hh 8 hours"`
  set dd_lst = `date +%d --date="$year$mm$dd $hh 8 hours"`
  set hh_lst = `date +%H --date="$year$mm$dd $hh 8 hours"`
  #
  set yy1 = `date +%Y --date="$year$mm$dd $hh 1 day ago"`
  set mm1 = `date +%m --date="$year$mm$dd $hh 1 day ago"`
  set dd1 = `date +%d --date="$year$mm$dd $hh 1 day ago"`
  set hh1 = `date +%H --date="$year$mm$dd $hh 1 day ago"`

# ---
# --- step 2 : create namelist.sc for successive correction
# ---          run successive correction
  
  if( ${if_bogus} == yes ) then
    set bogus = '.true.,'
  else
    set bogus = '.false.,'
  endif
  #
  if( ${verify_result} == yes ) then
    set verify = '.true.,'
  else
    set verify = '.false.,'
  endif

  echo ' Step 1.                   '
  echo ' Data preparing ....  '
  set time1 = ` date +%s `
  #
  if( ! -e ${bg_path}/bg_${year}${mm}${dd}_${hh}.nc ) then
    echo '   '
    echo ' No BG File, Please Check.'
    exit
  endif
  cp -f ${bg_path}/bg_${year}${mm}${dd}_${hh}.nc ./wrf_${year}${mm}${dd}_${hh}.nc
  if( ${verify_result} == yes && -e ${verify_path}/ver_${year}${mm}${dd}_${hh}.nc ) then 
    cp -f ${verify_path}/ver_${year}${mm}${dd}_${hh}.nc wrf_${year}${mm}${dd}_${hh}.nc_cfsr
  else
    echo ' No verify file, cannot verify   '
    set verify = '.false.,'
  endif
  #
  set sfc_type = 0
  set wsat_type = 0
  #
  if( -e ${obs_path}/${yy_lst}${mm_lst}${dd_lst}${hh_lst}.sfc ) then
    set sfc_type = 1
    cp -f ${obs_path}/${yy_lst}${mm_lst}${dd_lst}${hh_lst}.sfc .
  else
    echo ' No obs sfc file  '
    set sfc_type = 0
  endif
  #
  cp -f ${current_path}/data/mask.nc .
  #
  if( -e ${obs_path}/wsat_${year}${mm}${dd}v7 ) then
    cp -f ${obs_path}/wsat_${year}${mm}${dd}v7 .
    set wsat_type = 1
  else
    set wsat_type = 0
  endif
  if( ${hh} == 00 )then
    if( -e ${obs_path}/wsat_${yy1}${mm1}${dd1}v7 ) then
      cp -f ${obs_path}/wsat_${yy1}${mm1}${dd1}v7 .
      if( ${wsat_type} == 0 )then
        set wsat_type = 1
      endif
    endif
  endif
  cp -f ${current_path}/code/succ_*.gs .

  sed 's/BG_TIME/'${year}${mm}${dd}_${hh}'/g'  ./code/namelist.sc_base | \
  sed 's/LONBEG/'${lon_beg}'/g'                                   | \
  sed 's/LONEND/'${lon_end}'/g'                                   | \
  sed 's/LATBEG/'${lat_beg}'/g'                                   | \
  sed 's/LATEND/'${lat_end}'/g'                                   | \
  sed 's/RES/'${res}'/g'                                          | \
  sed 's/SFCTYPE/'${sfc_type}'/g'                                 | \
  sed 's/WSATTYPE/'${wsat_type}'/g'                               | \
  sed 's/SFC_TIME/'${yy_lst}${mm_lst}${dd_lst}${hh_lst}'/g'       | \
  sed 's/WSAT_TIME/'${year}${mm}${dd}'/g'                         | \
  sed 's/BG_DATE/'${year}${mm}${dd}${hh}'/g'                      | \
  sed 's/TRUE/'${bogus}'/g'                                       | \
  sed 's/TC_NUM/'${TC_num}'/g'                                    | \
  sed 's/TCLON/'${TC_lon}'/g'                                     | \
  sed 's/TCLAT/'${TC_lat}'/g'                                     | \
  sed 's/VMAX_RATIO/'${ratio}'/g'                                 | \
  sed 's/VMAX/'${vmax}'/g'                                        | \
  sed 's/RMAX/'${rmax}'/g'                                        | \
  sed 's/R50/'${R50}'/g'                                          | \
  sed 's/R30/'${R30}'/g'                                          | \
  sed 's/SCHEME/'${bogus_scheme}'/g'                              | \
  sed 's/VAR_SELECT/'${var_select}'/g'                            | \
  sed 's/WINDOW/'${window_time}'/g'                               | \
  sed 's/WIND_RA/'${wind_radius}'/g'                              | \
  sed 's/TEMP_RA/'${temp_radius}'/g'                              | \
  sed 's/VERIFY/'${verify}'/g'                                    | \
  sed 's/OBS_RA/'${obs_radius}'/g'                      > namelist.sc

  set time2 = ` date +%s `
  set cost_time = ` expr ${time2} - ${time1} `
  echo '    -- cost time = '${cost_time}' s'
  echo '     '
  #
  echo ' Step 2.                   '
  echo ' Successive correction (and bogus tc) ....  '
  set time1 = ` date +%s `
  ./code/su_correction.exe
  #./code/su_correction.exe > msg

  set time2 = ` date +%s `
  set cost_time = ` expr ${time2} - ${time1} `
  echo '    -- cost time = '${cost_time}' s'
  echo '     '

  if( ! -e original_uv.nc ) then
    echo '   '
    echo ' No Correction  Result, Please Check.'
    exit
  endif
#
# --- step 3 : Keep the results
# ---          

  echo ' Step 3.                   '
  echo ' Do ploting ....  '
  set time1 = ` date +%s `

  echo '    background '
  gradsnc -blc succ_before1.gs > msg_plot
  echo '    analysis '
  gradsnc -blc succ_after1.gs > msg_plot

  mkdir -p result/${year}${mm}${dd}_${hh}

  mv *.eps result/${year}${mm}${dd}_${hh}/
  mv *.jpg result/${year}${mm}${dd}_${hh}

  mv RMSE.dat result/${year}${mm}${dd}_${hh}
  set time2 = ` date +%s `
  set cost_time = ` expr ${time2} - ${time1} `
  echo '    -- cost time = '${cost_time}' s'
  echo '     '

# --- do clean
# ---
  rm -f mask.nc wsat*v7 *.sfc wrf*.nc wrf*_cfsr namelist.sc
  rm -f succ_*.gs msg_plot msg original_uv.nc successive_uv.nc
  mv fort.999 result/${year}${mm}${dd}_${hh}/run.msg

#========================== End of file ===============================
