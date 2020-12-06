#!/bin/bash

device=0
device1=1
device2=2
device3=3
specific=bowling_alley
module=layer3

# # default settings (first experiment group...)
# python3 compute_metrics.py -baseline --experiment sgd_final --device $device1 --specific $specific --module $module &
# python3 compute_metrics.py -debias --experiment sgd_final --device $device2 --specific $specific --module $module & 
# python3 compute_metrics.py -adv --experiment linear_adv_test --device $device3 --specific $specific --module $module
# # post trained versions...
#python3 compute_metrics.py -baseline -post --experiment sgd_final --device $device1 --specific $specific --module $module &
#python3 compute_metrics.py -debias -post --experiment sgd_final --device $device2 --specific $specific --module $module & 
#python3 compute_metrics.py -adv -post --experiment linear_adv_test --device $device3 --specific $specific --module $module

# actually compiling
#python3 compute_metrics.py -final --specific $specific --module $module
#python3 compute_metrics.py -final -post --specific $specific --module $module

# ~~~~~~~~~~~~~~~~~~~~~

#specific=runway

#python3 compute_metrics.py -baseline --experiment sgd_final --device $device1 --specific $specific --module $module &
#python3 compute_metrics.py -debias --experiment sgd_final --device $device2 --specific $specific --module $module & 
#python3 compute_metrics.py -adv --experiment linear_adv_test --device $device2 --specific $specific --module $module & 
#python3 compute_metrics.py -final --specific $specific --module $module

# ~~~~~~~~~~~~~~~~~~~~~

# specifics=("bowling_alley" "runway" "bamboo_forest")
# specifics=("bamboo_forest")
# specifics=("bowling_alley" "runway" "bamboo_forest" "bedroom" "corn_field")
# specifics=("bowling_alley" "runway" "bamboo_forest" "bedroom" "corn_field" "track" "ski_slope")
specifics=("bowling_alley" "runway" "bamboo_forest" "bedroom" "corn_field" "track" "ski_slope" "cockpit" "bus_interior" "laundromat")
# specifics=("bowling_alley")
epochs="5 10 15"
seeds="0 1 2"
declare specific;
printf -v specific "%s "  "${specifics[@]}"
ratios="0.0 0.25 0.5 0.75 1.0"
metrics="tcav" #projection leakage tcav" #leakage" #"accuracy projection tcav"

# python3 compute_metrics.py -fresh -post -baseline --experiment sgd_final --device $device --specific $specific --module $module --metrics $metrics --ratios $ratios --seeds $seeds &
# sleep 2
# python3 compute_metrics.py -fresh -post -adv --experiment linear_adv_test_stronger --adv_extra stronger --device $device1 --specific $specific --module $module --metrics $metrics --ratios $ratios --seeds $seeds &
# sleep 2
# python3 compute_metrics.py -fresh -post -debias --experiment sgd_final_0.05 --debias_extra 0.05 --device $device2 --specific $specific --module $module --metrics $metrics --ratios $ratios --seeds $seeds
# sleep 2
# sleep 2
# process results
# python3 compute_metrics.py -final -post --metrics accuracy projection --debias_extra 2000 --specific $specific --module $module --ratios $ratios --seeds 0
# python3 compute_metrics.py -fresh -post -adv --experiment linear_adv_test_stronger --adv_extra stronger --device $device1 --specific $specific --module $module --metrics accuracy --ratios $ratios --seeds $seeds
# GRAPHING
# : '
python3 compute_metrics.py -final -comparison -aggregated -post \
        --specifics bowling_alley bamboo_forest.bowling_alley.runway bamboo_forest.bedroom.bowling_alley.corn_field.runway bamboo_forest.bedroom.bowling_alley.corn_field.runway.ski_slope.track bamboo_forest.bedroom.bowling_alley.bus_interior.cockpit.corn_field.laundromat.runway.ski_slope.track \
        --ratios $ratios --seeds $seeds --debias_extra 0.05 --adv_extra stronger --module $module #  
# '
# python3 compute_metrics.py -final -comparison -aggregated -post \
#         --specifics bamboo_forest.bedroom.bowling_alley.bus_interior.cockpit.corn_field.laundromat.runway.ski_slope.track \
#         --ratios $ratios --seeds $seeds --debias_extra 0.05 --adv_extra stronger --module $module # bowling_alley bamboo_forest.bowling_alley.runway bamboo_forest.bedroom.bowling_alley.corn_field.runway bamboo_forest.bedroom.bowling_alley.corn_field.runway.ski_slope.track
# for c in 0 1 2 3 4 5 6 7 8 9
# do
#     python3 compute_metrics.py -post -final -comparison --class_ $c\
#         --specifics bowling_alley bamboo_forest.bowling_alley.runway bamboo_forest.bedroom.bowling_alley.corn_field.runway bamboo_forest.bedroom.bowling_alley.corn_field.runway.ski_slope.track bamboo_forest.bedroom.bowling_alley.bus_interior.cockpit.corn_field.laundromat.runway.ski_slope.track \
#         --ratios $ratios --seeds $seeds --debias_extra 0.05 --adv_extra stronger --module $module
# done

specifics=("bowling_alley" "runway" "bamboo_forest" "bedroom" "corn_field" "track" "ski_slope")
# specifics=("bowling_alley")
declare specific;
printf -v specific "%s "  "${specifics[@]}"
# python3 compute_metrics.py -fresh -post -baseline --experiment sgd_final --device $device --specific $specific --module $module --metrics $metrics --ratios $ratios --seeds $seeds &
# sleep 2
# python3 compute_metrics.py -fresh -post -adv --experiment linear_adv_test_stronger --adv_extra stronger --device $device1 --specific $specific --module $module --metrics $metrics --ratios $ratios --seeds $seeds &
# sleep 2
# python3 compute_metrics.py -fresh -post -debias --experiment sgd_final_0.05 --debias_extra 0.05 --device $device2 --specific $specific --module $module --metrics $metrics --ratios $ratios --seeds $seeds


# python3 compute_metrics.py -fresh -post -baseline --experiment sgd_final --device $device2 --specific $specific --module $module --metrics $metrics --ratios $ratios --seeds $seeds &
# sleep 2
# python3 compute_metrics.py -fresh -post -adv --experiment linear_adv_test --device $device3 --specific $specific --module $module --metrics $metrics --ratios $ratios --seeds $seeds
# sleep 2
# python3 compute_metrics.py -fresh -debias --experiment sgd_final_1000 --debias_extra 1000 --device $device2 --specific $specific --module $module --metrics $metrics --ratios $ratios --seeds $seeds #&
# python3 compute_metrics.py -final -post --metrics accuracy leakage projection tcav --debias_extra 1000 --specific $specific --module $module --ratios $ratios --seeds 0 1 2

# python3 compute_metrics.py -fresh -adv --experiment linear_adv_test --device $device3 --specific $specific --module $module --metrics $metrics --ratios $ratios --seeds 0 1 2 &

# process results
# python3 compute_metrics.py --metrics $metrics -final --specific $specific --module $module --ratios $ratios --seeds 0 1 2 &
# python3 compute_metrics.py -post -final --specific $specific --module $module --ratios 0.0 0.1 0.25 0.5 0.75 0.9 1.0 --seeds 0 1 2
# python3 compute_metrics.py -post --models 0 1 2 --debias_extra 100_linear --metrics $metrics -final --specific $specific --module $module --ratios 0.0 0.1 0.25 0.5 0.75 0.9 1.0 --seeds 0 1 2

# specifics=("bowling_alley" "runway" "bamboo_forest" "bedroom" "corn_field")
# declare specific;
# printf -v specific "%s "  "${specifics[@]}"
# metrics="accuracy" #projection tcav leakage" #"accuracy projection tcav"
# #python3 compute_metrics.py -fresh -debias --debias_extra 500 --experiment sgd_final_500 --device 6 --specific $specific --module $module --metrics $metrics --ratios 0.0 0.25 0.5 0.75 1.0 --seeds 0 1 2 &
# #sleep 2
# #python3 compute_metrics.py -fresh -debias --debias_extra 250 --experiment sgd_final_250 --device 7 --specific $specific --module $module --metrics $metrics --ratios 0.0 0.25 0.5 0.75 1.0 --seeds 0 1 2

# python3 compute_metrics.py --metrics $metrics --debias_extra 250 -final --specific $specific --module $module --ratios 0.0 0.25 0.5 0.75 1.0 --seeds 0 1 2


# ADDING AN EXTRA THING TO APPEND
# python3 compute_metrics.py -post -debias --epochs 9 --experiment sgd_final_100_jump --debias_extra 100_jump --device 0 --specific $specific --module $module --metrics projection --ratios $ratios --seeds 0&
# sleep 2
# python3 compute_metrics.py -post -debias --epochs 12 --experiment sgd_final_100_jump --debias_extra 100_jump --device 0 --specific $specific --module $module --metrics projection --ratios $ratios --seeds 0&
# sleep 2
# python3 compute_metrics.py -post -debias --epochs 15 --experiment sgd_final_100_jump --debias_extra 100_jump --device 1 --specific $specific --module $module --metrics projection --ratios $ratios --seeds 0&
# sleep 2
# python3 compute_metrics.py -post -debias --epochs 9 --experiment sgd_final_100_linear --debias_extra 100_linear --device 3 --specific $specific --module $module --metrics projection --ratios $ratios --seeds 0&
# sleep 2
# python3 compute_metrics.py -post -debias --epochs 12 --experiment sgd_final_100_linear --debias_extra 100_linear --device 4 --specific $specific --module $module --metrics projection --ratios $ratios --seeds 0&
# sleep 2
# python3 compute_metrics.py -post -debias --epochs 15 --experiment sgd_final_100_linear --debias_extra 100_linear --device 5 --specific $specific --module $module --metrics projection --ratios $ratios --seeds 0&

# python3 compute_metrics.py -post --models 0 1 2 --debias_extra 100_jump --metrics accuracy projection -final --specific $specific --module $module --ratios 0.0 0.25 0.5 0.75 1.0 --seeds 0 1 2
