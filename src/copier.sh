#!/bin/bash

id='13874.png'
cp /mnt/dev/datasets/gated2depth/data/real/gated0_10bit/"${id}" /mnt/dev/projects/edric/DPT_Gated_trianer/src/input_day/gated0_10bit/"${id}"

cp /mnt/dev/datasets/gated2depth/data/real/gated1_10bit/"${id}" /mnt/dev/projects/edric/DPT_Gated_trianer/src/input_day/gated1_10bit/"${id}"

cp /mnt/dev/datasets/gated2depth/data/real/gated2_10bit/"${id}" /mnt/dev/projects/edric/DPT_Gated_trianer/src/input_day/gated2_10bit/"${id}"

#cp -r  /home/xt/PycharmProjects/trianingModule/src/input_day/* /home/xt/PycharmProjects/trianingModule/input/
echo "complete!"
