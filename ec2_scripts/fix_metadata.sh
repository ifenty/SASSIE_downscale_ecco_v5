#!/bin/bash

p=`pwd`
eccho ${p}
cd $1
echo `pwd`
for i in *nc
do
  echo $i
  ncatted -a history,global,o,sng,'Initial release of the ECCO N1 Sassie Ocean-Sea Ice Simulation' -a metadata_link,global,o,sng,'https://cmr.earthdata.nasa.gov/search/collections.umm_json?ShortName=TBD' "${i}" -O

done

cd ${p}
echo `pwd`
