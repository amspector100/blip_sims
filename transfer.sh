# Reminder: add scripts/final before data for the new versions (some of 6/21 and all afterwards)
dir="lp_int_sol/"
date="2021-11-09"
origin="asherspector@login.rc.fas.harvard.edu:~/adaptive2/blip_sims/data/"$dir$date

destination="data/"$dir

echo $origin
echo $destination

scp -r $origin $destination
