## Instructions to recreate COCO Experiments

To train from scratch, copy over the contents of scripts folder to MetaOrthogonalization/coco:

` cp scripts/* ./` <br>

Next, prepare the COCO dataset by running 

` ./prepare_coco.sh ` <br>

Then, run `./coco_pipeline.sh`. Follow the comments in the scripts to choose which subset of experiments you want to run. <br>

To download the pretrained models, go to this Drive: 
