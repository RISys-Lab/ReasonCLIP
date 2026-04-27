#!/bin/bash

# Released model list for eval scripts.
# All ReasonCLIP/ReasonSigLIP release repos include their own processor files.
models=(
  "openai/clip-vit-large-patch14-336"
  "fesvhtr/RC-L14-336-S0-Rea"
  "fesvhtr/RC-L14-336-S0-Des"
  "fesvhtr/RC-L14-336-S1"
  "fesvhtr/RC-L14-336-S2"

  "openai/clip-vit-base-patch32"
  "fesvhtr/RC-B32-S0-Rea"
  "fesvhtr/RC-B32-S0-Des"
  "fesvhtr/RC-B32-S1"
  "fesvhtr/RC-B32-S2"
  "fesvhtr/RC-B32-READ"

  "openai/clip-vit-large-patch14"
  "fesvhtr/RC-L14-224-S0-Rea"
  "fesvhtr/RC-L14-224-S0-Des"
  "fesvhtr/RC-L14-224-S1"
  "fesvhtr/RC-L14-224-S2"

  "google/siglip-so400m-patch14-384"
  "fesvhtr/RS-So14-S0-Rea"
  "fesvhtr/RS-So14-S0-Des"
  "fesvhtr/RS-So14-S1"
  "fesvhtr/RS-So14-S2"

  "google/siglip2-so400m-patch14-384"
  "fesvhtr/RS2-So14-S0-Rea"
  "fesvhtr/RS2-So14-S0-Des"
  "fesvhtr/RS2-So14-S1"
  "fesvhtr/RS2-So14-S2"

  "google/siglip2-giant-opt-patch16-384"
  "fesvhtr/RS2-GO16-S0-Rea"
  "fesvhtr/RS2-GO16-S0-Des"
  "fesvhtr/RS2-GO16-S1"
  "fesvhtr/RS2-GO16-S2"
)
