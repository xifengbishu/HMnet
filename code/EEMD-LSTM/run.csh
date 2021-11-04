#!/bin/csh
echo " EEMD MODEL, run.csh 20 5"
echo " SOLO MODEL, run.csh 20 999"

set SITES = 5
set IMF = 5
set epoch = 1000
# ----------------------------------
# ----------------------------------
# ----------------------------------
set nsite = 0
while ( ${nsite} < $SITES )
set nimf = 3
while ( ${nimf} < $IMF )
   python EEMD_LSTM.py $nsite $nimf > msg
set nimf = `expr ${nimf} + 1`
end
   python EEMD_LSTM.py $nsite 999 >> msg
set nsite = `expr ${nsite} + 1`
end

exit

# ----------------------------------
# ----------------------------------
# ----------------------------------
set nsite = 0
while ( ${nsite} < $SITES )
   python Multi_step.py $nsite
   mkdir RESULT$nsite
   mv *jpg RESULT$nsite
   mv *txt RESULT$nsite
set nsite = `expr ${nsite} + 1`
end
exit

