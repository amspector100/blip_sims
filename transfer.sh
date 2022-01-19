# Reminder: add scripts/final before data for the new versions (some of 6/21 and all afterwards)
dir="${2:-glms/}"
date="${1:-2022-01-03}"
echo "Date: $date, Dir: $dir"

origin="asherspector@login.rc.fas.harvard.edu:~/adaptive2/blip_sims/data/"$dir$date

destination="data/"$dir

echo $origin
echo $destination

scp -r $origin $destination
