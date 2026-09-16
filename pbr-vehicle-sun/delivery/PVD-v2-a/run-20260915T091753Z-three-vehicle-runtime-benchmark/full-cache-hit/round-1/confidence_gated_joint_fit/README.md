# SSE confidence-gated multi-vehicle fit

Each vehicle is fitted independently with the same camera-visible contour, robust bidirectional distance, P95 percentile and optional line-structure settings. Only vehicles passing both hard gates enter the weighted and shared-angle joint outputs. If none pass, `status` is `no_valid_sun_information`.

Each shadow input is the official SSISv2 shadow member associated with that vehicle and is consumed directly without source-mask postprocessing.

Every `vehicle_XX/fit` directory contains best-fit evidence. Independent candidate sheets are optional; the shared `joint_fit` retains the requested candidate sheet. All visualizations are regenerated with the exact objective used for selection.
