* ----------------------------------------------
* --- Purpose:
* ---
* -----------------------------------------------
  'sdfopen ./original_uv.nc'

* --- multiple cases or single case ?
* --- m: multiple ( default )
* --- s: single   ( should be declared explicitly! )
* --- "case_no = 4" means "open ../case4/*.ctl", for single only!
  case_type   = 's'
  case_no     = 1 
* --- For example, never use it unless you modify this script!
* --- Be careful !
  my_own_plot = 0

* --- convert time from UTC to LST ?
* --- If the time is already LST, then no!
* --- UTC2LST = 1, yes ; 0, no
  UTC2LST = 0

* --- arrangement of page and pannels
* --- xy_ratio can be calculated aumomatically
* --- However, you can tune its value for your own aim
* ---
  page_type  = 'l'
  x_num      = 1
  y_num      = 1
  ratio_auto = 1
* --- Normally, do not use it.
  if( ratio_auto = 0 )
    xy_ratio  = 0.8
  endif

* --- It is ok for normal use
* --- Normally, do not need to change it.
  if( page_type = 'p' ) 
    x_sta =  0.05
    x_end =  8.15
    y_beg = 10.90
  endif
  if( page_type = 'l' ) 
    x_sta =  0.05
    x_end = 10.50
    y_beg =  8.30
  endif
 
* ---  reset_domain = 1 / 0
* ---  1: yes ; 0: no (default domain is used)
  reset_domain = 0
  lon_domain   = 'lon  40 160'
  lat_domain   = 'lat   0  60'

* --- draw labels of longitude and latitude along axises
* --- xlint_num =5 means an interval of 5 deg. along X-axis
  string_size  = 0.13
  xlint_num    = 10
  ylint_num    = 10
* 
* --- draw color bar ( 1: yes ; 0: no )
* --- size of color bar string at the bottom
* --- height of color bar
  color_bar    = 1
  bar_str_size = 0.13
  bar_height   = 0.16
  color_set    = 4
  var_unit     = '`ao`nC'
* --- examples
* var_unit     = '`ao`nC'   celsius degree
* var_unit     = 'K'        K degree
* var_unit     = 'm'        meter
* var_unit     = 'mm'       precipitation

* --- wind vector/barb at 10m above sea level
* --- wind_plot ( 1: yes  ; 0: no )
* --- wind_type ( 1: barb ; 2: vector )
* --- wind_colr ( 0: white; 1: black; any other color! )
* --- full barb ( 4: 4m/s ; 5: 5m/s )
  wind_plot = 1
  wind_type = 1
  wind_colr = 1
* --- wind_type: barb
  if( wind_type = 1 ) 
    interval  = 9
    barb_size = 0.04
    full_barb = 4
  endif
* --- wind_type: vector
  if( wind_type = 2 )
    interval  = 8
    arrowhead = 0.04
*   --- arrow length: length
*   --- this length present the speed scale (m/s)
    length    = 0.2
    scale     = 10
  endif

* --- set number of output picture (pic_num)
* --- set case name
  pic_num   = 2
  case_name = 'orginal'
* --- force pic_num =1 when case_type='s'
  if( case_type = 's')
    pic_num = 1
  endif 

* --- time setting    ( tt = t_beg + (n-1)*t_int )
* --- begining times  ( t_beg )
* --- times interval  ( t_int )
* --- times can be given one by one (times_given=1)
* --- 1: give  ; 0: aumomatically calculated
* --- picture 1
    t_beg1 = 1
    t_int1 = 1
    times_given1 = 0
    times1 = ' 2 5 7 9 11 13 15 17 '
* --- picture 2
    t_beg2 = 11
    t_int2 = 2
    times_given2 = 0
    times2 = ' 11 13 15 17 19 '

* --- along X direction
* --- title number
    title1 = ' a b c d e f g h i j k l m n o p '
    title2 = ' e f g h i j k l'
* --- along Y direction
* --- experiment case name 
  case = ' A B C D E F '

* --- Change size of the title of Date-time & panel number
* --- title_resize is between 1.0 and 1.3
* --- It is normally 1.0 
  title_resize = 1.00

* --- the panels below the first row can be without
* --- date-time titles, but with panel number only !
* --- 1: with ; 0: without 
  time_title = 0

* =======================================
* ---- End of Your Own Modifications ----
* ---- Do not touch anything below.  ----
* =======================================

* --- Big Loop
 page_num = 1
 while( page_num <= pic_num )
* --- set page infomation
   if( page_num = 1 )
     t_beg = t_beg1       
     t_int = t_int1
     title = title1
     times = times1
     times_given = times_given1
   endif
   if( page_num = 2 )
     t_beg = t_beg2       
     t_int = t_int2
     title = title2
     times = times2
     times_given = times_given2
   endif

*
* ---- Step 1: check input
*
  'q gxinfo'
  rec  = sublin(result,2)
  xsiz = subwrd(rec,4)
  ysiz = subwrd(rec,6)
  if( xsiz = 8.5 & page_type = 'l' )
    say ' ======================================= '
    say ' '
    say ' Please grads -blc plot_fogtop_wrf.gs !  '
    say ' '
    say ' ======================================= '
    quit
  endif
  if( xsiz = 11 & page_type = 'p' )
    say ' ======================================= '
    say ' '
    say ' Please grads -bpc plot_fogtop_wrf.gs !  '
    say ' '
    say ' ======================================= '
    quit
  endif

*
* ---- Step 2: calculate xy_ratio 
* ----         longitude and latitude
  if( reset_domain = 1 )
    'set 'lon_domain
    'set 'lat_domain
  endif
  if( ratio_auto = 1 )
    'display lon'
    'query gxinfo'
    line3=sublin(result,3)
    line4=sublin(result,4)
    tmpa=subwrd(line3,4)
    tmpb=subwrd(line3,6)
    tmpc=subwrd(line4,4)
    tmpd=subwrd(line4,6)
    xy_ratio=(tmpd-tmpc)/(tmpb-tmpa)
    'clear'
  endif
* ---- longitude and latitude
  'q dims'
  lined2 = sublin( result, 2)
  lined3 = sublin( result, 3)
  xlon1 = subwrd(lined2, 6)
  xlon2 = subwrd(lined2, 8)
  ylat1 = subwrd(lined3, 6)
  ylat2 = subwrd(lined3, 8)

*
* ---- Step 3: define page settings for all panels
*
  x_int = ( x_end - x_sta ) / x_num
  y_int = xy_ratio * x_int
  y_sta = y_beg - y_num*y_int
  'set vpage 0 'x_int' 0 'y_int''
  x_v = subwrd(result,5) - 0.01
  y_v = subwrd(result,6) - 0.01
* ----------
  x  = x_sta
  m  = 1
  while( m <= x_num )
    y = y_beg - y_int
    n = 1
    while( n <= y_num )
      _vpage.m.n = ' set vpage 'x' 'x+x_int' 'y' 'y+y_int
      _parea.m.n = ' set parea 0 'x_v' 0 'y_v
*     --- for later use ---
      no = ( n - 1 ) * x_num + m      
      _vpg.no = ' set vpage 'x' 'x+x_int' 'y' 'y+y_int
      _par.no = ' set parea 0 'x_v' 0 'y_v
*     --------------------- 
      y = y - y_int
      n = n + 1
    endwhile
    x  =  x + x_int
    m  =  m + 1
  endwhile

*
* ---- Step 4: start to plot panel one by one
*
* === For your own case !
*     from left to right, from up to down
*     1   2  3
*     4   5  6
*     7   8  9
*
 if( my_own_plot = 1 )
   n = 1
   while( n <= y_num )
    m = 1
    while( m <= x_num )
*     --- start to plot panels one by one
      number = ( n - 1 ) * x_num + m  
      _vpg.number      
      _par.number
*     --- control panels one by one
      if( number = 1 )

      endif
      if( number = 2 )

      endif
      if( number = 3 )

      endif
      add_frame()
      'set vpage off '
      'set parea off '
      m = m + 1
    endwhile
    n = n + 1
   endwhile
 endif
 
*
* === For Sea Fog (WRF/RAMS) USE only below !
* 
  tt = t_beg
  m  = 1       
  while( m <= x_num )
    say ' === Page='page_num'; column ='m
* ----------
    n = 1       
    while( n <= y_num )
      say ' === Page='page_num'; row ='n
*     --- set virtual page
      _vpage.m.n
      _parea.m.n
*     --- time settings
*     --- multiple cases
      if( times_given = 1 ) 
        tt_given = subwrd(times,m)
        'set t 'tt_given
      else
        'set t 'tt
      endif
*     --- single case
      if( case_type = 's' )
        ms = (n-1) * x_num + m
        if( times_given = 1 )
          tt_given = subwrd(times,ms)
        else
          tt_given = t_beg + (ms-1)*t_int   
        endif
        'set t 'tt_given
      endif
*     ---
*     --- plot setting   
      'set xlint 'xlint_num
      'set ylint 'ylint_num
      'set grid on 2 1'
      'set grads off'
      'set font 1'
*     --- reset domain
      if( reset_domain = 1 ) 
        'set 'lon_domain
        'set 'lat_domain
      endif
*     --- determine case_n, multiple or single case ?
      case_n = n
      if( case_type = 's' )
        case_n = case_no
      endif
*
* ======== Start to plot variabels ========
*
      'set mpdraw off'
      'set gxout shaded'
      'set clevs 0 1'
      'set ccols 0 15 0'
      'd landmask'

      'set gxout shaded'
*      'set clevs -10 -8 -6 -4 -2 0 2 4 6 8 10'
      do_color_set(color_set)
      'set rbcols '_ncolor
      'set clevs '_ncvalue
      'd maskout(t2-273.16,landmask*(-1.))'
 
*     --- set map  
      'set map 1 1 4'
        
*     --- plot U, V
      if( wind_plot = 1 ) 
        'set ccolor 'wind_colr
        if( wind_type = 1 ) 
          'set gxout barb'
          'set digsiz 'barb_size
          blen = 10. / full_barb
        endif
        if( wind_type = 2 )
          'set gxout vector'
          'set arrowhead 'arrowhead
          'set arrscl 'length' 'scale
          'set arrlab off'
          blen = 1
        endif
        intv = interval
        u_part = 'skip(maskout(u10*'blen',landmask*(-1)),'intv')'
        v_part = 'skip(maskout(v10*'blen',landmask*(-1)),'intv')'
        'd 'u_part' ; 'v_part
      endif
   
*      'set gxout shaded'
*      'set clevs -10 -8 -6 -4 -2 0 2 4 6 8 10'
*      do_color_set(color_set)
*      'set rbcols '_ncolor
*      'd mag(su,sv)'
*
* ======== Stop ploting variabels ========
*

*     --- draw  LST
      date_time = get_date_time ( UTC2LST )
*
*     --- calculate size settings for titles
*
      y1 = 0.65 * title_resize
      x2 = 6.00 * title_resize
      strsize1 = 0.40 * title_resize
      str_x    = 0.24 * title_resize
      str_y    = 0.52 * title_resize
*
      yy1 = 1.40 * title_resize
      yy2 = 0.65 * title_resize
      xx1 = 1.30 * title_resize
      strsize2 = 0.50 * title_resize
      str_xx1  = 0.70 * title_resize
      str_xx2  = 0.24 * title_resize
      str_xx3  = 0.47 * title_resize
      str_yy   = 1.25 * title_resize
*
*     ---- first row
      if( n = 1 ) 
        'set line 0'
*        'draw recf 0.02 'y_v - y1' 'x2' 'y_v
        'set line 1 1 6'
*        'draw  rec 0.02 'y_v - y1' 'x2' 'y_v
        'set strsiz 'strsize1
*        'draw string 'str_x' 'y_v - str_y' 'date_time
      else
*     ---- from the second row
        if( time_title = 1 )
          'set line 0'
          'draw recf 0.02 'y_v - y1' 'x2' 'y_v
          'set line 1 1 6'
          'draw  rec 0.02 'y_v - y1' 'x2' 'y_v
          'set strsiz 'strsize1
          'draw string 'str_x' 'y_v - str_y' 'date_time
        endif
      endif
*     --- draw plot number
      if( time_title = 0  ) 
        yy1 = yy1 - yy2
        str_yy = str_yy - yy2
        yy2 = 0
      endif
      'set line 0'
      'draw recf 0.02 'y_v - yy1' 'xx1' 'y_v - yy2
      'set line 1 1 6'
      'draw rec  0.02 'y_v - yy1' 'xx1' 'y_v - yy2
      'set strsiz 'strsize2
      if( case_type = 'm' )
        case_num  = subwrd(case,n)
        title_num = subwrd(title,m)
        'draw string 'str_xx1'  'y_v - str_yy' 'case_num
        'draw string 'str_xx2'  'y_v - str_yy' 'title_num
      endif
      if( case_type = 's' )
        ms = (n-1) * x_num + m
        title_num = subwrd(title,ms)
        'draw string 'str_xx3'  'y_v - str_yy' 'title_num
      endif

* --- add a bold frame around the old one
      add_frame()
*
      'set vpage off '
      'set parea off '
      n = n + 1
    endwhile
*   -------------
    tt = tt + t_int
    m  =  m + 1
  endwhile

**************************************************
*****  End of LOOP for plotting panels !!! *******
**************************************************


*
* ---- Step 5
*
* === plot longtitude and latitude labels
* --- define labels along axises.
* --- longitude
  nloop = 1
  nxlab = 0
  while( nloop < 1000 )
    xlab = ( nloop - 1 ) * xlint_num
    if( xlab >= (xlon1+0.2) & xlab <= (xlon2-0.2) )
      if( nxlab = 0 )
        xlab_beg = xlab 
      endif 
      nxlab = nxlab + 1
    endif
    nloop = nloop + 1
  endwhile
* --- latitude
  nloop = 1
  nylab = 0
  while( nloop < 1000 )
    ylab = ( nloop - 1 ) * ylint_num - 90
    if( ylab > (ylat1+0.2) & ylab < (ylat2-0.2) )
      if( nylab = 0 )
        ylab_beg = ylab
      endif
      nylab = nylab + 1
    endif
    nloop = nloop + 1
  endwhile
* --- check
*  say xlab_beg'  'ylab_beg
*  say nxlab'  'nylab


* --- longitude
  y_lon = y_sta - string_size * 0.5
  x  = x_sta
  m  = 1
  while( x <= x_end - x_int )
* ----------
    nx = 1
    xlen = xlon2 - xlon1
    while( nx <= nxlab )
      xlon = xlab_beg + (nx-1) * xlint_num
      pos_cent = ( xlon - xlon1 ) / xlen
      pos = x + pos_cent * x_int
     'set string 1 tc'
     'set strsiz 'string_size
      if( xlon <= 180 )
        'draw string 'pos' 'y_lon' 'xlon'E'
      else
         xlonw = 360 - xlon
        'draw string 'pos' 'y_lon' 'xlonw'W'
      endif
      nx = nx + 1
    endwhile
*   -------------
    x  =  x + x_int
    m  =  m + 1
  endwhile

* --- latitude
  x_lat = x_end + string_size * 1.40 + 0.05
  y = y_beg - y_int
  n = 1
  while( y >= y_sta )
* ----------
    ny = 1
    ylen = ylat2 - ylat1
    while( ny <= nylab )
      ylat = ylab_beg + (ny-1) * ylint_num
      pos_cent = ( ylat - ylat1 ) / ylen
      pos = y + pos_cent * y_int
     'set string 1 c'
     'set strsiz 'string_size
      if( ylat < 0 )
        ylats = ylat * (-1)
        'draw string 'x_lat' 'pos' 'ylats'S'
      endif
      if( ylat = 0 )
        'draw string 'x_lat' 'pos' EQ'
      endif
      if( ylat > 0 )
        'draw string 'x_lat' 'pos' 'ylat'N'
      endif
      ny = ny + 1
    endwhile
*   -------------
    y = y - y_int
    n = n + 1
  endwhile

*
* ---- Step 6
*
* === plot color bar 
* --- dependent on x_end and x_sta
* --- bar_length_percent = 0.55

* --- start to draw color bar below

  if( page_type = 'p' )
    percent = 0.65 
  endif
  if( page_type = 'l' )
    percent = 0.55
  endif
  delx = x_end - x_sta
  half_deltx = delx * 0.5 
  bar_length = delx * percent
  x_beg  = half_deltx - bar_length  * 0.5
  x_tail = x_beg + bar_length

  if( color_bar = 1 )
* --- draw bar only
* --- calculate nbar = 13 ?
  nbar_num = 1
  nbar = 0
  while( nbar_num < 40 )
    val = subwrd( _ncvalue , nbar_num )    
    if( val = '' ) 
    else
      nbar = nbar + 1
    endif
    nbar_num = nbar_num + 1
  endwhile

  x0 = x_beg
  y0 = y_sta - (0.1+bar_height) - string_size - 0.1
  m  = 1
  while( m <= nbar+1 )
    col_bar = subwrd( _ncolor, m )
    if( m = 1 )
      hei_bar = ''
    else
      hei_bar = subwrd( _ncvalue, m-1 )
    endif
    'set line 'col_bar' 1'
    xint  = (x_tail - x_beg)/nbar
    xbar1 = x0 + (m-1)*xint
    xbar2 = xbar1 + xint
    if( m <= nbar+1 )
      'draw recf 'xbar1' 'y0' 'xbar2' 'y0+bar_height
      'set line 1 1 3'
      'draw rec 'xbar1' 'y0' 'xbar2' 'y0+bar_height
    endif
*   --- draw string
   'set string 1 tc'
   'set strsiz 'bar_str_size
   if( hei_bar != '' & mod(m, 2) = 1 ) 
     'draw string 'xbar1' 'y0-0.05'  'hei_bar
   endif
*   --- next
    m  =  m + 1
  endwhile
*   --- draw unit
  'set strsiz 'bar_str_size
  'draw string 'xbar1+0.25' 'y0-0.05' 'var_unit

  endif
* --- stop drawing color bar here

*
* ---- Step 7
*
* --- draw vector scale right to color bar
  if( wind_plot =1 & wind_type = 2 )
    len = length * ( x_int / xsiz )
    x   = x_tail + 1.0
    y   = y_sta -  0.15 - string_size 
    if( color_bar = 1 )
      y = y0 + bar_height * 0.30 
    endif
    'set line 1 1 4'
    'draw line 'x-len/2.' 'y' 'x+len/2.' 'y
    'draw line 'x+len/2.-0.05' 'y+0.03' 'x+len/2.' 'y
    'draw line 'x+len/2.-0.05' 'y-0.03' 'x+len/2.' 'y
    'set string 1 c'
    'set strsiz 'bar_str_size
    'draw string 'x' 'y-bar_str_size-0.05' 'scale' m/s'
  endif

*
* ---- Step 8
*
* --- output name
  if( pic_num >= 2 ) 
    outname = case_name''page_num
  else
    outname = case_name
  endif
*
* --- output *.gm pictures
*   'printim ./'outname'.gif gif x600 y800 white'
  'enable print ./'outname'.gm'
  'print'
  'disable print'
* --- convert gm to eps 
  '! gxeps -R -c -i 'outname'.gm -o 'outname'.eps'

* --- convert gm to png
* --- opengrads needed
  '! rm -f 'outname'.gm'

* --- convert eps to gif 
* --- imagick software needed
  '! convert -density 576 -resize 25% -trim 'outname'.eps 'outname'.jpg'
*  '! rm -f 'outname'.bmp' 

* --- Big Loop End
    'clear'
    'reset'
    page_num = page_num + 1
  endwhile

* --- exit GrADS
  quit

*================== function definitions ==================
* 
*
function incdtgh(dtgh,inc)
*
*  increment a dtg by inc hours
*  RESTRICTIONS!!
*  (1)  inc > 0
*  (2)  inc < 24
*
  moname = 'JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC'
  monday = '31 28 31 30 31 30 31 31 30 31 30 31'

  iyr = substr(dtgh,1,2)*1
  imo = substr(dtgh,3,2)*1
  ida = substr(dtgh,5,2)*1
  ihr = substr(dtgh,7,2)*1

  if( mod(iyr,4) = 0 )
    monday = '31 29 31 30 31 30 31 31 30 31 30 31'
  endif

  ihr = ihr + inc
*  say 'ihr = 'ihr
*  say 'ida = 'ida

  if( ihr >= 24 )
    ihr = ihr - 24
    ida = ida + 1
  endif

*  say 'new ihr = 'ihr' new ida = 'ida' imo = 'imo
  if( ida > subwrd(monday,imo) )
    ida = ida - subwrd(monday,imo)
*    say 'inside check ida = 'ida' monday = 'subwrd(monday,imo)
    imo = imo+1
  endif

  if( ida <= 0 )
    imo = imo - 1
    ida = subwrd(monday,imo) - ida + 1
  endif

  if( imo >= 13 )
    imo = imo - 12
    iyr = iyr + 1
  endif

  if( imo < 10 ); imo = '0'imo; endif
  if( ida < 10 ); ida = '0'ida; endif
  if( ihr < 10 ); ihr = '0'ihr; endif

*return (iyr%imo%ida%ihr)
* --- Gao changed
  mon = subwrd(moname,imo)
return (ihr%'00LST '%ida%' '%mon)

function get_date_time ( args )
  'q time'
  b = subwrd(result,3)
  UTC2LST = subwrd(args,1) 
  if( UTC2LST = 0 )
    uhour = substr(b,1,2)
    udate = substr(b,4,2)
    month = substr(b,6,3)
    date_time=uhour%'00LST '%udate%' '%month
  endif
  if( UTC2LST = 1 )
    mon='JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC'
    yy = substr(b,11,2)
    mm = substr(b,6,3)
    dd = substr(b,4,2)
    hh = substr(b,1,2)
    im = 1
    while ( im <= 12 )
      mo=subwrd( mon, im )
      if( mo = mm )
        mt = im
        if( mt < 10 )
          mt = '0'mt
        endif
        break
      endif
      im = im + 1
    endwhile
    dtgh = yy%mt%dd%hh
    inc  = 8
    date_time = incdtgh(dtgh,inc)
  endif
return(date_time)


function mod( i0, inc )
  if( inc!=0 )
    imod = int(i0/inc)
  else
    imod = int(i0/1)
  endif
  imod = i0 - imod*inc
return(imod)

function int(i0)
  i = 0
  while( i < 12 )
    i = i + 1
    if( substr(i0,i,1) = '.' )
      i0 = substr(i0, 1, i-1)
      break
    endif
  endwhile
return(i0)

function add_frame()
  'q gxinfo'
  line3 = sublin(result,3)
  line4 = sublin(result,4)
  x1 = subwrd(line3,4)
  x2 = subwrd(line3,6)
  y1 = subwrd(line4,4)
  y2 = subwrd(line4,6)
  'set line 1 1 6'
  'draw rec 'x1' 'y1' 'x2' 'y2
return

function do_color_set(args)
*
* --- You can add other color sets for your own use.
*
  color_num = subwrd(args,1) 
*
* --- 1: blue and red ( for difference)
  if( color_num = 1 ) 
    'set rgb 16   0   0 255'
    'set rgb 17  55  55 255'
    'set rgb 18 110 110 255'
    'set rgb 19 165 165 255'
    'set rgb 20 220 220 255'
    'set rgb 21 255 220 220'
    'set rgb 22 255 165 165'
    'set rgb 23 255 110 110'
    'set rgb 24 255  55  55'
    'set rgb 25 255   0   0'
    'set rgb 26 160   0   0'
    _ncolor  = '16  17  18  19  20  0  21 22 23 24 25 26'
    _ncvalue = '-5  -4  -3  -2  -1  0  1  2  3  4  5'
  endif
* --- 2: gray, red and green ( for seafog top )
  if( color_num = 2 )
    'set rgb 79 130 130 130'
    'set rgb 80 150 150 150'
    'set rgb 81 170 170 170'
    'set rgb 82 190 190 190'
    'set rgb 83 255 225 230'
    'set rgb 84 255 189 200'
    'set rgb 85 255 157 172'
    'set rgb 86 255 120 141'
    'set rgb 87 255  60  90'
    'set rgb 88 220   0  30'
    'set rgb 89 156 235 176'
    'set rgb 90  94 220 125'
    'set rgb 91  37 175  70'
    'set rgb 92  29 137  56'
    'set rgb 99  82  85 209'
    _ncolor  = '79  80  81  82  83  84  85  86  87  88  89  90  91 92'
    _ncvalue = '50 100 150 200 250 300 350 400 450 500 550 600 650'
  endif
* --- 3: gray, blue and green ( for seafog top )
  if( color_num = 3 )
    'set rgb 79 130 130 130'
    'set rgb 80 150 150 150'
    'set rgb 81 170 170 170'
    'set rgb 82 190 190 190'
    'set rgb 83 220 220 247'
    'set rgb 84 200 200 233'
    'set rgb 85 180 180 219'
    'set rgb 86 160 160 205'
    'set rgb 87 140 140 191'
    'set rgb 88 120 120 177'
    'set rgb 92 156 235 176'
    'set rgb 91  94 220 125'
    'set rgb 90  37 175  70'
    'set rgb 89  29 137  56'
    'set rgb 99 255   0   0'
    _ncolor  = ' 79  80  81  82  83  84  85  86  87  88  89  90  91  92  99'
    _ncvalue = ' -3   0   3   6   9  12  15  18  21  24  27  30  33  36'
  endif
* -----
* --- 4: blue and red ( for inversion )
  if( color_num = 4 )
    'set rgb  16    0    0   30'
    'set rgb  17    4    4   37'
    'set rgb  18    8    8   44'
    'set rgb  19   12   12   51'
    'set rgb  20   16   16   58'
    'set rgb  21   20   20   65'
    'set rgb  22   24   24   72'
    'set rgb  23   28   28   79'
    'set rgb  24   32   32   86'
    'set rgb  25   36   36   93'
    'set rgb  26   40   40  100'
    'set rgb  27   44   44  107'
    'set rgb  28   48   48  114'
    'set rgb  29   52   52  121'
    'set rgb  30   56   56  128'
    'set rgb  31   60   60  135'
    'set rgb  32   70   70  142'
    'set rgb  33   80   80  149'
    'set rgb  34   90   90  156'
    'set rgb  35  100  100  163'
    'set rgb  36  110  110  170'
    'set rgb  37  120  120  177'
    'set rgb  38  130  130  184'
    'set rgb  39  140  140  191'
    'set rgb  40  150  150  198'
    'set rgb  41  160  160  205'
    'set rgb  42  170  170  212'
    'set rgb  43  180  180  219'
    'set rgb  44  190  190  226'
    'set rgb  45  200  200  233'
    'set rgb  46  210  210  240'
    'set rgb  47  220  220  247'
    'set rgb  48  255  250  205'
    'set rgb  49  255  247  185'
    'set rgb  50  255  244  165'
    'set rgb  51  255  241  145'
    'set rgb  52  255  238  125'
    'set rgb  53  255  226  113'
    'set rgb  54  255  214  101'
    'set rgb  55  255  202   89'
    'set rgb  56  255  190   77'
    'set rgb  57  255  178   65'
    'set rgb  58  255  166   53'
    'set rgb  59  255  154   41'
    'set rgb  60  255  142   29'
    'set rgb  61  255  130   17'
    'set rgb  62  255  118    5'
    'set rgb  63  255  106    0'
    'set rgb  64  255   94    0'
    'set rgb  65  255   82    0'
    'set rgb  66  255   70    0'
    'set rgb  67  255   58    0'
    'set rgb  68  255   46    0'
    'set rgb  69  255   34    0'
    'set rgb  70  235   24    0'
    'set rgb  71  215   14    0'
    'set rgb  72  195    4    0'
    'set rgb  73  175    0    0'
    'set rgb  74  155    0    0'
    'set rgb  75  135    0    0'
    'set rgb  76  115    0    0'
    'set rgb  77   95    0    0'
    'set rgb  78   75    0    0'
    'set rgb  79   55    0    0'
    'set rgb  80   30    0    0'
    'set rgb  81   10    0    0'
    _ncolor  = '  31 33 35 37 39 41 43 45 47 48 52 55 58 61 64 67 70 73 75 77 79 81 '
    _ncvalue = '  -4 -2  0  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38'
  endif
* ----
  if( color_num = 5 )
    'set rgb 16   0   0 255'
    'set rgb 17  55  55 255'
    'set rgb 18 110 110 255'
    'set rgb 19 165 165 255'
    'set rgb 20 220 220 255'
    'set rgb 21 255 220 220'
    'set rgb 22 255 165 165'
    'set rgb 23 255 110 110'
    'set rgb 24 255  55  55'
    'set rgb 25 255   0   0'
    'set rgb 26 160   0   0'
    _ncolor  = '18  19  20  21 22 23 24 25 26'
    _ncvalue = ' 0  2  4  6  8  10  12  14 '
  endif
* -----
return


*
* ======================= End of this gs file ============================
