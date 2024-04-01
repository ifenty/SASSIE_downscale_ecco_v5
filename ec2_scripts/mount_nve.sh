#!/bin/bash

# one integer argument how many nvme disks are on the machine
# new directories go to 
# /nvme_dataNN

for i in `seq 1 ${1}`
do

   sudo file -s /dev/nvme${i}n1
   sudo mkfs -t xfs /dev/nvme${i}n1
   sudo mkdir /nvme_data${i}
   sudo mount /dev/nvme${i}n1 /nvme_data${i}
   sudo chmod ugo+wx /nvme_data${i} 
done

