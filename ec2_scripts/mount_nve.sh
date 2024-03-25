#!/bin/bash

# one integer argument how many nvme disks are on the machine
for i in `seq 1 ${1}`
do

   sudo file -s /dev/nvme${i}n1
   sudo mkfs -t xfs /dev/nvme${i}n1
   sudo mkdir /data${i}
   sudo mount /dev/nvme${i}n1 /data${i}
   sudo chmod ugo+wx /data${i} 
done

